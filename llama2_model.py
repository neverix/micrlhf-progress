from typing import List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
from jax.sharding import Mesh, NamedSharding, Sharding, PartitionSpec
from state_util import dummy_stateful, dummy_caching


class ModuleList(eqx.Module):
    modules: List[eqx.Module]
    
    out_shape: Tuple[int]

    def __init__(self, modules: List[eqx.Module]):
        self.modules = modules
        self.out_shape = (self.modules[-1].out_shape[0],)

    def __call__(self, *args, **kwargs):
        for module in self.modules:
            args, kwargs = module(*args, **kwargs)
        return args, kwargs
    
    def __getitem__(self, key):
        return self.modules[key]
    
    def __iter__(self):
        return iter(self.modules)


class Weight(eqx.Module):
    weight: jax.Array
    policy: jmp.Policy
    out_shape: Tuple[int]

    def __init__(
        self,
        shape,
        sharding: Sharding,
        policy: jmp.Policy = jmp.get_policy("float32")
    ):
        self.policy = policy
        self.weight = jax.device_put(
            jnp.empty(shape, dtype=policy.compute_dtype),
            sharding,
        )
        self.out_shape = (self.weight.shape[1],)

    @dummy_caching
    @dummy_stateful
    def __call__(self, x):
        x = self.policy.cast_to_compute(x)
        return x @ self.weight


class Embedding(eqx.Module):
    weight: jax.Array

    vocab_size: int
    hidden_size: int
    
    out_shape: Tuple[int]

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
        
        self.out_shape = (self.hidden_size,)

    @dummy_caching
    @dummy_stateful
    def __call__(self, x):
        return self.weight[x]


class LayerNorm(eqx.Module):
    weight: jax.Array

    hidden_size: int
    rms_norm_eps: float = 1e-5
    
    out_shape: Tuple[int]

    def __init__(self, hidden_size, mesh: Mesh):
        self.hidden_size = hidden_size
        self.weight = jax.device_put(
            jnp.empty((self.hidden_size,), dtype=jnp.float32),
            NamedSharding(mesh, spec=PartitionSpec(None)),
        )
        
        self.out_shape = (self.hidden_size,)

    @dummy_caching
    @dummy_stateful
    def __call__(self, x):
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.square(x).mean(axis=-1, keepdims=True))
        x = self.weight * x / (rms + self.rms_norm_eps)
        x = x.astype(orig_dtype)
        return x


class MLP(eqx.Module):
    up_proj: jax.Array
    gate_proj: jax.Array
    down_proj: jax.Array
    policy: jmp.Policy

    intermediate_size: int
    hidden_size: int
    
    out_shape: Tuple[int]

    def __init__(self, hidden_size, intermediate_size, mesh: Mesh, policy: jmp.Policy):
        self.policy = policy
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        up_sharding = NamedSharding(mesh, spec=PartitionSpec(None, "mp"))
        down_sharding = NamedSharding(mesh, spec=PartitionSpec("mp", None))
        self.up_proj = Weight(
            (self.hidden_size, self.intermediate_size), up_sharding, policy,
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
        
        self.out_shape = (self.hidden_size,)

    def __call__(self, x, state, cache):
        x = self.policy.cast_to_compute(x)
        gate, state, cache = self.gate_proj(x, state, cache)
        up, state, cache = self.up_proj(x, state, cache)
        x = gate * jax.nn.sigmoid(up)
        x, state, cache = self.down_proj(x, state, cache)
        return x, state, cache


class RotaryEmbedding(eqx.Module):
    inv_freq: jax.Array
    
    out_shape: Tuple[int]

    def __init__(self, hidden_size, mesh: Mesh):
        self.inv_freq = jax.device_put(
            1.0 / jax.lax.rsqrt(10000 ** (jnp.arange(0, hidden_size, 2) / hidden_size)),
            NamedSharding(mesh, spec=PartitionSpec(None)),
        )
        
        self.out_shape = (hidden_size,)

    @dummy_caching
    @dummy_stateful
    def __call__(self, x):
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)

        sequence = jnp.arange(x.shape[0])
        angles = jnp.einsum("i,j->ij", sequence, jax.lax.stop_gradient(self.inv_freq))
        sins = jnp.sin(angles)
        coss = jnp.cos(angles)
        half, odd = jnp.split(x, 2, axis=-1)
        x = jnp.concatenate(
            [half * coss + odd * sins, half * sins - odd * coss], axis=1
        )

        return x.astype(orig_dtype)


class Attention(eqx.Module):
    hidden_size: int
    out_shape: Tuple[int]

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.out_shape = (self.hidden_size,)

    @dummy_caching
    @dummy_stateful
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
    
    out_shape: Tuple[int]

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
            self.hidden_size // self.num_attention_heads, mesh,
        )
        self.attention = Attention(self.hidden_size // self.num_attention_heads)

        qkv_sharding = NamedSharding(mesh, spec=PartitionSpec(None, "mp"))
        o_sharding = NamedSharding(mesh, spec=PartitionSpec("mp", None))
        self.q_proj = Weight((self.hidden_size, self.hidden_size), qkv_sharding, policy,)
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
        self.o_proj = Weight((self.hidden_size, self.hidden_size), o_sharding, policy,)
        self.out_shape = (self.hidden_size,)

    def __call__(self, x, state: eqx.nn.State, cache: dict, attention_mask=None):
        # for now, regular attention
        x = self.policy.cast_to_compute(x)

        q, state, cache = jax.vmap(self.q_proj)(x, state, cache)
        k, state, cache = jax.vmap(self.k_proj)(x, state, cache)
        v, state, cache = jax.vmap(self.v_proj)(x, state, cache)

        remb_k = jax.vmap(self.rotary_emb, in_axes=(-2, None, -2), out_axes=(-2, None, -2))
        remb_q = jax.vmap(remb_k, in_axes=(-2, None, -2), out_axes=(-2, None, -2))
        q = q.reshape(q.shape[:-1] + (self.num_key_value_heads, self._group_size, -1))
        k = k.reshape(k.shape[:-1] + (self.num_key_value_heads, -1))
        # vmap will try to split the cache
        q, state, cache_ = remb_q(q, state, {})
        cache = {**cache, **cache_}
        k, state, cache_ = remb_k(k, state, {})
        cache = {**cache, **cache_}
        v = v.reshape(v.shape[:-1] + (self.num_key_value_heads, -1))

        o, state, cache = self.attention(q, k, v, state, cache, attention_mask=attention_mask)
        o = o.reshape(o.shape[:-3] + (-1,))
        o, state, cache = self.o_proj(o, state, cache)

        return o, state, cache


class LLaMALayer(eqx.Module):
    input_layernorm: LayerNorm
    mlp: MLP
    self_attn: SelfAttention
    post_attention_layernorm: LayerNorm
    
    out_shape: Tuple[int]

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
        self.mlp = MLP(hidden_size, intermediate_size, mesh, policy,)
        self.self_attn = SelfAttention(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            mesh,
            policy,
        )
        self.post_attention_layernorm = LayerNorm(hidden_size, mesh)
        self.out_shape = (hidden_size,)

    def __call__(self, x, state: eqx.nn.State, cache: dict):
        y, state, cache = jax.vmap(self.input_layernorm)(x, state, cache)
        y, state, cache = self.self_attn(y, state, cache)
        x = x + y
        y, state, cache = jax.vmap(self.post_attention_layernorm)(x, state, cache)
        y, state, cache = jax.vmap(self.mlp)(y, state, cache)
        x = x + y
        return x, state, cache


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
    
    out_shape: Tuple[int]

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

        layers = []
        self.num_layers = num_layers
        for i in range(self.num_layers):
            layers.append(
                LLaMALayer(
                    self.hidden_size,
                    self.intermediate_size,
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    mesh,
                    policy,
                )
            )
        self.layers = ModuleList(layers)

        self.norm = LayerNorm(self.hidden_size, mesh,)
        
        self.out_shape = (self.hidden_size,)

    def __call__(self, x, state: eqx.nn.State, cache: dict):
        x, state, cache = jax.vmap(self.embed_tokens, in_axes=(0, None, 0), out_axes=(0, None, 0))(x, state, cache)
        x = self.policy.cast_to_compute(x)
        for layer in self.layers:
            x, state, cache = layer(x, state, cache)
        # x = jax.lax.scan(lambda carry, layer: layer(carry), x, self.layers)[0]
        x, state, cache = jax.vmap(self.norm, in_axes=(0, None, 0), out_axes=(0, None, 0))(x, state, cache)
        return x, state, cache


class LLaMA(eqx.Module):
    policy: jmp.Policy
    lm_head: Embedding
    model: LLaMAModel

    vocab_size: int = 32_000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    num_layers: int = 32
    
    out_shape: Tuple[int]

    def __init__(self, mesh: Mesh, policy: jmp.Policy = jmp.get_policy("float32"),):
        self.model = LLaMAModel(
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.num_layers,
            mesh,
            policy=policy,
        )
        self.lm_head = Weight(
            (self.hidden_size, self.vocab_size),
            NamedSharding(mesh, spec=PartitionSpec(None, None)),
            policy=policy,
        )
        self.policy = policy
        self.out_shape = (self.vocab_size,)

    def __call__(self, x, state: eqx.nn.State, cache: dict):
        embeds, state, cache = self.model(x, state, cache)
        return jax.vmap(self.lm_head, in_axes=(0, None, 0))(embeds, state, cache)
