from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp


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
        ) * ((1 / self.hidden_size) ** 0.5)

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
        rms = jnp.square(x).mean(axis=-1, keepdims=True).sqrt()
        return self.weight * x / (rms + self.rms_norm_eps)


class MLP(eqx.Module):
    up_proj: jax.Array
    gate_proj: jax.Array
    down_proj: jax.Array

    intermediate_size: int
    hidden_size: int

    def __init__(self, hidden_size, intermediate_size, key):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        up_proj_key, key = jax.random.split(key)
        gate_proj_key, key = jax.random.split(key)
        down_proj_key, key = jax.random.split(key)

        self.up_proj = jax.random.normal(
            up_proj_key, (self.hidden_size, self.intermediate_size)
        ) / (self.hidden_size**0.5)
        self.gate_proj = jax.random.normal(
            gate_proj_key, (self.hidden_size, self.intermediate_size)
        ) / (self.hidden_size**0.5)
        self.down_proj = jax.random.normal(
            down_proj_key, (self.intermediate_size, self.hidden_size)
        ) / (self.intermediate_size**0.5)

    def __call__(self, x):
        y = x @ self.gate_proj * jax.nn.sigmoid(x @ self.up_proj)
        return x + y @ self.down_proj


class RotaryEmbedding(eqx.Module):
    inv_freq: jax.Array

    def __init__(self, hidden_size, key):
        self.inv_freq = jax.lax.rsqrt(
            10000 ** (2 * jnp.arange(hidden_size // 2) / hidden_size)
        )

    def __call__(self, x, axis=-2):
        x = x[:, None, :] * self.inv_freq[None, :, None]
        x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
        return x


class SelfAttention(eqx.Module):
    q_proj: jax.Array
    k_proj: jax.Array
    v_proj: jax.Array
    o_proj: jax.Array
    rotary_emb: RotaryEmbedding

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    _group_size: int

    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads, key):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self._group_size = self.num_attention_heads // self.num_key_value_heads

        self.rotary_emb = RotaryEmbedding(hidden_size, key)
        q_proj_key, key = jax.random.split(key)
        k_proj_key, key = jax.random.split(key)
        v_proj_key, key = jax.random.split(key)
        o_proj_key, key = jax.random.split(key)

        self.q_proj = jax.random.normal(
            q_proj_key, (self.hidden_size, self.hidden_size)
        ) / (self.hidden_size**0.5)
        self.k_proj = jax.random.normal(
            k_proj_key, (self.hidden_size, self.hidden_size // self._group_size)
        ) / (self.hidden_size**0.5)
        self.v_proj = jax.random.normal(
            v_proj_key, (self.hidden_size, self.hidden_size // self._group_size)
        ) / (self.hidden_size**0.5)
        self.o_proj = jax.random.normal(
            o_proj_key, (self.hidden_size, self.hidden_size)
        ) / (self.hidden_size**0.5)

    def __call__(self, x):
        # for now, regular attention
        q = x @ self.q_proj
        k = x @ self.k_proj
        v = x @ self.v_proj

        q = q.reshape(
            q.shape[:-2] + (q.shape[1], self.num_key_value_heads, self._group_size, -1)
        )
        q = self.rotary_emb(q, axis=-3)
        k = k.reshape(
            k.shape[:-2]
            + (
                k.shape[1],
                self.num_key_value_heads,
                self.hidden_size // self.num_key_value_heads // self._group_size,
            )
        )
        k = self.rotary_emb(k, axis=-3)
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
        o = jnp.einsum("...abhg,...bhd->...ahgd", attention_matrix, v) @ self.o_proj

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
    ):
        mlp_key, key = jax.random.split(key)
        attn_key, ln_key = jax.random.split(key)
        input_ln_key, post_ln_key = jax.random.split(ln_key)

        self.input_layernorm = LayerNorm(hidden_size, input_ln_key)
        self.mlp = MLP(hidden_size, intermediate_size, mlp_key)
        self.self_attn = SelfAttention(
            hidden_size, num_attention_heads, num_key_value_heads, attn_key
        )
        self.post_attention_layernorm = LayerNorm(hidden_size, post_ln_key)

    def __call__(self, x):
        y = self.input_layernorm(x)
        y = self.post_attention_layernorm(self.self_attn(y))
        y = self.mlp(y)
        return x + y


class LLaMAModel(eqx.Module):
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
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

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
                )
            )

    def __call__(self, x):
        x = self.embed_tokens(x)
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

    def __init__(self, key):
        lm_head_key, key = jax.random.split(key)
        self.lm_head = Embedding(self.hidden_size, key=lm_head_key, is_unembed=True)
        self.model = LLaMAModel(
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            key,
        )

    def __call__(self, x):
        return self.lm_head(self.model(x))
