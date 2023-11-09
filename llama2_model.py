from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
from jax.sharding import Mesh, NamedSharding, Sharding, PartitionSpec
from jax.debug import print as jdp


class Weight(eqx.Module):
    weight: jax.Array
    policy: jmp.Policy

    def __init__(
        self,
        shape,
        sharding: Sharding,
        policy: jmp.Policy = jmp.get_policy("float32"),
    ):
        self.policy = policy
        self.weight = jax.device_put(
            jnp.empty(shape, dtype=policy.compute_dtype),
            sharding,
        )

    def __call__(self, x):
        x = self.policy.cast_to_compute(x)
        return x @ self.weight


class Embedding(eqx.Module):
    weight: jax.Array

    vocab_size: int
    hidden_size: int

    def __init__(self, vocab_size, hidden_size, mesh: Mesh):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weight = jax.device_put(
            jnp.empty(
                (self.vocab_size, self.hidden_size),
                dtype=jnp.float32,
            ),
            NamedSharding(mesh, spec=PartitionSpec(None, None)),
        )

    def __call__(self, x):
        return self.weight[x]


class LayerNorm(eqx.Module):
    weight: jax.Array

    hidden_size: int
    rms_norm_eps: float = 1e-5

    def __init__(self, hidden_size, mesh: Mesh):
        self.hidden_size = hidden_size
        self.weight = jax.device_put(
            jnp.empty((self.hidden_size,), dtype=jnp.float32),
            NamedSharding(mesh, spec=PartitionSpec(None)),
        )

    def __call__(self, x):
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.square(x).mean(axis=-1, keepdims=True))
        x = self.weight * x / (rms + self.rms_norm_eps)
        return x.astype(orig_dtype)


class MLP(eqx.Module):
    up_proj: jax.Array
    gate_proj: jax.Array
    down_proj: jax.Array
    policy: jmp.Policy

    intermediate_size: int
    hidden_size: int

    def __init__(self, hidden_size, intermediate_size, mesh: Mesh, policy: jmp.Policy):
        self.policy = policy
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        up_sharding = NamedSharding(mesh, spec=PartitionSpec(None, "mp"))
        down_sharding = NamedSharding(mesh, spec=PartitionSpec("mp", None))
        self.up_proj = Weight(
            (self.hidden_size, self.intermediate_size), up_sharding, policy
        )
        self.gate_proj = Weight(
            (self.hidden_size, self.intermediate_size),
            up_sharding,
            policy,
        )
        self.down_proj = Weight(
            (self.intermediate_size, self.hidden_size),
            down_sharding,
            policy,
        )

    def __call__(self, x):
        x = self.policy.cast_to_compute(x)
        x = self.gate_proj(x) * jax.nn.sigmoid(self.up_proj(x))
        return self.down_proj(x)


class RotaryEmbedding(eqx.Module):
    inv_freq: jax.Array

    def __init__(self, hidden_size, mesh: Mesh):
        self.inv_freq = jax.device_put(
            1.0 / jax.lax.rsqrt(10000 ** (jnp.arange(0, hidden_size, 2) / hidden_size)),
            NamedSharding(mesh, spec=PartitionSpec(None)),
        )

    def __call__(self, x):
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)

        sequence = jnp.arange(x.shape[0])
        angles = jnp.einsum("i,j->ij", sequence, jax.lax.stop_gradient(self.inv_freq))
        sins = jnp.sin(angles)
        coss = jnp.cos(angles)
        half, odd = jnp.split(x, 2, axis=1)
        x = jnp.concatenate(
            [half * coss + odd * sins, half * sins - odd * coss], axis=1
        )

        return x.astype(orig_dtype)


class Attention(eqx.Module):
    hidden_size: int

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def __call__(self, q, k, v, attention_mask=None):
        attention_matrix = jnp.einsum("...ahgd,...bhd->...abhg", q, k)
        attention_matrix = attention_matrix / (self.hidden_size**0.5)
        if attention_mask is None:
            attention_mask = jnp.tril(
                jnp.ones_like(attention_matrix[..., 0, 0], dtype=jnp.bool_)
            )[..., None, None]
        attention_matrix = jnp.where(attention_mask, attention_matrix, jnp.NINF)
        attention_matrix = jax.nn.softmax(attention_matrix, axis=-3)
        o = jnp.einsum("...abhg,...bhd->...ahgd", attention_matrix, v)
        return o


class SelfAttention(eqx.Module):
    q_proj: jax.Array
    k_proj: jax.Array
    v_proj: jax.Array
    o_proj: jax.Array

    rotary_emb: RotaryEmbedding
    attention: Attention

    policy: jmp.Policy

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    _group_size: int

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        mesh: Mesh,
        policy: jmp.Policy = jmp.get_policy("float32"),
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        assert self.num_attention_heads >= self.num_key_value_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0
        self._group_size = self.num_attention_heads // self.num_key_value_heads
        self.policy = policy

        self.rotary_emb = RotaryEmbedding(
            self.hidden_size // self.num_attention_heads, mesh
        )
        self.attention = Attention(self.hidden_size // self.num_attention_heads)

        qkv_sharding = NamedSharding(mesh, spec=PartitionSpec(None, "mp"))
        o_sharding = NamedSharding(mesh, spec=PartitionSpec("mp", None))
        self.q_proj = Weight((self.hidden_size, self.hidden_size), qkv_sharding, policy)
        self.k_proj = Weight(
            (self.hidden_size, self.hidden_size // self._group_size),
            qkv_sharding,
            policy,
        )
        self.v_proj = Weight(
            (self.hidden_size, self.hidden_size // self._group_size),
            qkv_sharding,
            policy,
        )
        self.o_proj = Weight((self.hidden_size, self.hidden_size), o_sharding, policy)

    def __call__(self, x, attention_mask=None):
        # for now, regular attention
        x = self.policy.cast_to_compute(x)

        q = jax.vmap(self.q_proj)(x)
        k = jax.vmap(self.k_proj)(x)
        v = jax.vmap(self.v_proj)(x)

        remb_k = jax.vmap(self.rotary_emb, in_axes=-2, out_axes=-2)
        remb_q = jax.vmap(remb_k, in_axes=-2, out_axes=-2)
        q = q.reshape(q.shape[:-1] + (self.num_key_value_heads, self._group_size, -1))
        k = k.reshape(k.shape[:-1] + (self.num_key_value_heads, -1))
        q = remb_q(q)
        k = remb_k(k)
        v = v.reshape(v.shape[:-1] + (self.num_key_value_heads, -1))

        o = self.attention(q, k, v, attention_mask=attention_mask)
        o = o.reshape(o.shape[:-3] + (-1,))
        o = self.o_proj(o)

        return o


class LLaMALayer(eqx.Module):
    input_layernorm: LayerNorm
    mlp: MLP
    self_attn: SelfAttention
    post_attention_layernorm: LayerNorm

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        mesh: Mesh,
        policy: jmp.Policy = jmp.get_policy("float32"),
    ):
        self.input_layernorm = LayerNorm(hidden_size, mesh)
        self.mlp = MLP(hidden_size, intermediate_size, mesh, policy)
        self.self_attn = SelfAttention(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            mesh,
            policy,
        )
        self.post_attention_layernorm = LayerNorm(hidden_size, mesh)

    def __call__(self, x):
        x = x + self.self_attn(jax.vmap(self.input_layernorm)(x))
        x = x + jax.vmap(self.mlp)(jax.vmap(self.post_attention_layernorm)(x))
        return x


class LLaMAModel(eqx.Module):
    policy: jmp.Policy
    embed_tokens: Embedding
    norm: LayerNorm
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_layers: int
    layers: List[LLaMALayer]

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        num_layers,
        mesh: Mesh,
        policy: jmp.Policy = jmp.get_policy("float32"),
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.policy = policy

        self.embed_tokens = Embedding(self.vocab_size, self.hidden_size, mesh)

        self.layers = []
        self.num_layers = num_layers
        for i in range(self.num_layers):
            self.layers.append(
                LLaMALayer(
                    self.hidden_size,
                    self.intermediate_size,
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    mesh,
                    policy,
                )
            )

        self.norm = LayerNorm(self.hidden_size, mesh)

    def __call__(self, x):
        x = jax.vmap(self.embed_tokens)(x)
        x = self.policy.cast_to_compute(x)
        for layer in self.layers:
            x = layer(x)
        # x = jax.lax.scan(lambda carry, layer: layer(carry), x, self.layers)[0]
        x = jax.vmap(self.norm)(x)
        return x


class LLaMA(eqx.Module):
    lm_head: Embedding
    model: LLaMAModel

    vocab_size: int = 32_000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    num_layers: int = 32

    def __init__(self, mesh: Mesh, policy: jmp.Policy = jmp.get_policy("float32")):
        self.model = LLaMAModel(
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.num_layers,
            mesh,
            policy,
        )
        self.lm_head = Weight(
            (self.hidden_size, self.vocab_size),
            NamedSharding(mesh, spec=PartitionSpec(None, None)),
            policy,
        )

    def __call__(self, x):
        return jax.vmap(self.lm_head)(self.model(x))
