from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
from jax.sharding import Mesh, NamedSharding, Sharding, PartitionSpec


class Weight(eqx.Module):
    weight: jax.Array
    policy: jmp.Policy

    def __init__(
        self,
        shape,
        key,
        sharding: Sharding,
        policy: jmp.Policy = jmp.get_policy("float32"),
    ):
        self.policy = policy
        self.weight = jax.lax.with_sharding_constraint(
            policy.cast_to_param(jax.random.normal(key, shape) / (shape[-1] ** 0.5)),
            sharding,
        )

    def __call__(self, x):
        x = self.policy.cast_to_compute(x)
        return x @ self.weight


class Embedding(eqx.Module):
    weight: jax.Array

    vocab_size: int = 32_000
    hidden_size: int
    is_unembed: bool

    def __init__(self, hidden_size, key, is_unembed=False):
        self.hidden_size = hidden_size
        self.is_unembed = is_unembed
        self.weight = jax.random.normal(
            key, (self.vocab_size, self.hidden_size)[:: (-1 if is_unembed else 1)]
        )

    def __call__(self, x):
        if self.is_unembed:
            return x @ self.weight
        else:
            return self.weight[x]


class LayerNorm(eqx.Module):
    weight: jax.Array

    hidden_size: int
    rms_norm_eps: float = 1e-5

    def __init__(self, hidden_size, key):
        self.hidden_size = hidden_size
        self.weight = jax.random.normal(key, (self.hidden_size,))

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

    def __init__(
        self, hidden_size, intermediate_size, key, mesh: Mesh, policy: jmp.Policy
    ):
        self.policy = policy
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        up_proj_key, key = jax.random.split(key)
        gate_proj_key, key = jax.random.split(key)
        down_proj_key, key = jax.random.split(key)

        up_sharding = NamedSharding(mesh, spec=PartitionSpec(None, "mp"))
        down_sharding = NamedSharding(mesh, spec=PartitionSpec("mp", None))
        self.up_proj = Weight(
            (self.hidden_size, self.intermediate_size), up_proj_key, up_sharding, policy
        )
        self.gate_proj = Weight(
            (self.hidden_size, self.intermediate_size),
            gate_proj_key,
            up_sharding,
            policy,
        )
        self.down_proj = Weight(
            (self.intermediate_size, self.hidden_size),
            down_proj_key,
            down_sharding,
            policy,
        )

    def __call__(self, x):
        x = self.policy.cast_to_compute(x)
        y = self.gate_proj(x) * jax.nn.sigmoid(self.up_proj(x))
        return x + self.down_proj(y)


class RotaryEmbedding(eqx.Module):
    inv_freq: jax.Array

    def __init__(self, hidden_size):
        self.inv_freq = 1.0 / jax.lax.rsqrt(
            10000 ** (jnp.arange(0, hidden_size, 2) / hidden_size)
        )

    def __call__(self, x, seq_axis=-2, feature_axis=-1):
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)

        sequence = jnp.arange(x.shape[seq_axis])
        angles = jnp.einsum("i,j->ij", sequence, self.inv_freq)
        new_shape = [1] * len(x.shape)
        new_shape[seq_axis] = x.shape[seq_axis]
        new_shape[feature_axis] = x.shape[feature_axis] // 2
        angles = angles.reshape(new_shape)
        sins = jnp.sin(angles)
        coss = jnp.cos(angles)
        half, odd = jnp.split(x, 2, axis=feature_axis)
        x = jnp.concatenate(
            [half * coss - odd * sins, half * sins + odd * coss], axis=feature_axis
        )

        return x.astype(orig_dtype)


class SelfAttention(eqx.Module):
    q_proj: jax.Array
    k_proj: jax.Array
    v_proj: jax.Array
    o_proj: jax.Array
    policy: jmp.Policy
    rotary_emb: RotaryEmbedding

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    _group_size: int

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        key,
        mesh: Mesh,
        policy: jmp.Policy = jmp.get_policy("float32"),
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self._group_size = self.num_attention_heads // self.num_key_value_heads
        self.policy = policy

        per_group_size = self.hidden_size // self._group_size
        self.rotary_emb = RotaryEmbedding(per_group_size)
        q_proj_key, key = jax.random.split(key)
        k_proj_key, key = jax.random.split(key)
        v_proj_key, key = jax.random.split(key)
        o_proj_key, key = jax.random.split(key)

        qkv_sharding = NamedSharding(mesh, spec=PartitionSpec(None, "mp"))
        o_sharding = NamedSharding(mesh, spec=PartitionSpec("mp", None))
        self.q_proj = Weight(
            (self.hidden_size, self.hidden_size), q_proj_key, qkv_sharding, policy
        )
        self.k_proj = Weight(
            (self.hidden_size, per_group_size),
            k_proj_key,
            qkv_sharding,
            policy,
        )
        self.v_proj = Weight(
            (self.hidden_size, per_group_size),
            v_proj_key,
            qkv_sharding,
            policy,
        )
        self.o_proj = Weight(
            (self.hidden_size, self.hidden_size), o_proj_key, o_sharding, policy
        )

    def __call__(self, x):
        # for now, regular attention
        x = self.policy.cast_to_compute(x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(
            q.shape[:-2] + (q.shape[1], self.num_key_value_heads, self._group_size, -1)
        )
        q = self.rotary_emb(q, seq_axis=-4, feature_axis=-3)
        k = k.reshape(
            k.shape[:-2]
            + (
                k.shape[1],
                self.num_key_value_heads,
                self.hidden_size // self.num_key_value_heads // self._group_size,
            )
        )
        k = self.rotary_emb(k, seq_axis=-4, feature_axis=-3)
        v = k.reshape(
            v.shape[:-2]
            + (
                v.shape[1],
                self.num_key_value_heads,
                self.hidden_size // self.num_key_value_heads // self._group_size,
            )
        )

        attention_matrix = jnp.einsum("...ahgd,...bhd->...abhg", q, k)
        attention_matrix = attention_matrix / (self.hidden_size**0.5)
        attention_matrix = jax.nn.softmax(attention_matrix, axis=-2)
        o = jnp.einsum("...abhg,...bhd->...ahgd", attention_matrix, v)
        o = o.reshape(o.shape[:-3] + (o.shape[-3] * o.shape[-2] * o.shape[-1],))
        o = self.o_proj(o)

        return x + o


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
        key,
        mesh: Mesh,
        policy: jmp.Policy = jmp.get_policy("float32"),
    ):
        mlp_key, key = jax.random.split(key)
        attn_key, ln_key = jax.random.split(key)
        input_ln_key, post_ln_key = jax.random.split(ln_key)

        self.input_layernorm = LayerNorm(hidden_size, input_ln_key)
        self.mlp = MLP(hidden_size, intermediate_size, mlp_key, mesh, policy)
        self.self_attn = SelfAttention(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            attn_key,
            mesh,
            policy,
        )
        self.post_attention_layernorm = LayerNorm(hidden_size, post_ln_key)

    def __call__(self, x):
        y = self.input_layernorm(x)
        y = self.post_attention_layernorm(self.self_attn(y))
        y = self.mlp(y)
        return x + y


class LLaMAModel(eqx.Module):
    policy: jmp.Policy
    embed_tokens: Embedding
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    layers: List[LLaMALayer]

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        key,
        mesh: Mesh,
        policy: jmp.Policy = jmp.get_policy("float32"),
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.policy = policy

        embed_key, key = jax.random.split(key)
        self.embed_tokens = Embedding(self.hidden_size, embed_key)

        self.layers = []
        for i in range(24):
            layer_key, key = jax.random.split(key)
            self.layers.append(
                LLaMALayer(
                    self.hidden_size,
                    self.intermediate_size,
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    layer_key,
                    mesh,
                    policy,
                )
            )

    def __call__(self, x):
        x = self.embed_tokens(x)
        x = self.policy.cast_to_compute(x)
        for layer in self.layers:
            x = layer(x)
        return x


class LLaMA(eqx.Module):
    lm_head: Embedding
    model: LLaMAModel

    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_key_value_heads: int = 32

    def __init__(self, key, mesh: Mesh, policy: jmp.Policy = jmp.get_policy("float32")):
        lm_head_key, key = jax.random.split(key)
        self.lm_head = Embedding(self.hidden_size, key=lm_head_key, is_unembed=True)
        self.model = LLaMAModel(
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            key,
            mesh,
            policy,
        )

    def __call__(self, x):
        return self.lm_head(self.model(x))
