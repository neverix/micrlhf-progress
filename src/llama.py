# pretty much copied from https://github.com/google-deepmind/penzai/blob/main/penzai/example_models/gemma/model_core.py
import dataclasses

import jax
import jax.numpy as jnp
from penzai import pz  # ez


@dataclasses.dataclass
class LlamaConfig:
    vocab_size: int = 32_000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 128
    num_layers: int = 32
    parameter_dtype: jax.typing.DTypeLike = jnp.bfloat16
    activation_dtype: jax.typing.DTypeLike = jnp.float16


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class LlamaMLP(pz.nn.Sequential):
    @classmethod
    def from_config(cls, embedding_size: int, intermediate_size: int, dtype: jax.typing.DTypeLike):
        return cls([
            pz.nn.BranchAndMultiplyTogether(
                    branches=[
                    pz.nn.NamedGroup(
                        "gate",
                        [
                            pz.nn.add_parameter_prefix(
                                "gate_proj",
                                pz.nn.Linear.from_config(
                                    input_axes={"embedding": embedding_size},
                                    output_axes={"neurons": intermediate_size},
                                    dtype=dtype,
                                ),
                            ),
                            pz.nn.Elementwise(jax.nn.silu),
                        ],
                    ),
                    pz.nn.add_parameter_prefix(
                        "up_proj",
                        pz.nn.Linear.from_config(
                            input_axes={"embedding": embedding_size},
                            output_axes={"neurons": intermediate_size},
                            dtype=dtype,
                        ),
                    )
                ]
            ),
            pz.nn.add_parameter_prefix(
                "out_proj",
                pz.nn.Linear.from_config(
                    input_axes={"neurons": intermediate_size},
                    output_axes={"embedding": embedding_size},
                    dtype=dtype,
                ),
            ),
        ])


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class LlamaAttention(pz.nn.Attention):
  @classmethod
  def from_config(cls, config: LlamaConfig) -> "LlamaAttention":
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    assert num_heads % num_kv_heads == 0
    assert num_heads >= num_kv_heads
    q_rep = num_heads // num_kv_heads
    hidden_size = config.hidden_size
    projection_dim = config.head_dim

    return cls(
        input_to_query=pz.nn.Sequential([
            pz.nn.add_parameter_prefix(
                "query",
                pz.nn.Linear.from_config(
                    input_axes={"embedding": hidden_size},
                    output_axes={
                        "kv_heads": num_heads,
                        "q_rep": q_rep,
                        "projection": projection_dim,
                    },
                    dtype=config.parameter_dtype,
                ),
            ),
            pz.nn.ApplyRoPE.from_config(
                positions_tag="token_positions",
                embedding_axis="projection",
            ),
            pz.nn.ConstantRescale(
                by=jnp.array(
                    projection_dim**-0.5, dtype=config.activation_dtype
                )
            ),
        ]),
        input_to_key=pz.nn.Sequential([
            pz.nn.add_parameter_prefix(
                "key",
                pz.nn.Linear.from_config(
                    input_axes={"embedding": hidden_size},
                    output_axes={
                        "kv_heads": num_heads,
                        "projection": projection_dim,
                    },
                    dtype=config.parameter_dtype,
                ),
            ),
            pz.nn.ApplyRoPE.from_config(
                positions_tag="token_positions",
                embedding_axis="projection",
            ),
        ]),
        input_to_value=pz.nn.Sequential([
            pz.nn.add_parameter_prefix(
                "value",
                pz.nn.Linear.from_config(
                    input_axes={"embedding": hidden_size},
                    output_axes={
                        "kv_heads": num_heads,
                        "projection": projection_dim,
                    },
                    dtype=config.parameter_dtype,
                ),
            ),
        ]),
        query_key_to_attn=pz.nn.Sequential([
            pz.nn.NamedEinsum(
                (
                    {"seq": "tq", "kv_heads": "h", "q_rep": "r", "projection": "p"},
                    {"seq": "tkv", "kv_heads": "h", "projection": "p"},
                ),
                {"seq": "tq", "kv_heads": "h", "q_rep": "r", "kv_seq": "tkv"},
            ),
            pz.nn.ApplyAttentionMask.from_config(
                mask_tag="attn_mask",
                masked_out_value=jnp.array(
                    # ⁉️
                    -2.3819763e38, dtype=config.activation_dtype
                ),
            ),
            pz.nn.Softmax("kv_seq"),
        ]),
        attn_value_to_output=pz.nn.Sequential([
            pz.nn.NamedEinsum(
                (
                    {"seq": "tq", "kv_heads": "h", "q_rep": "r", "kv_seq": "tkv"},
                    {"seq": "tkv", "kv_heads": "h", "projection": "p"},
                ),
                {"seq": "tq", "kv_heads": "h", "q_rep": "r", "projection": "p"},
            ),
            pz.nn.add_parameter_prefix(
                "output",
                pz.nn.Linear.from_config(
                    input_axes={
                        "q_rep": q_rep,
                        "kv_heads": num_heads,
                        "projection": projection_dim,
                    },
                    output_axes={"embedding": hidden_size},
                    dtype=config.parameter_dtype,
                ),
            ),
        ]),
    )


def main():
    config = LlamaConfig()
    mlp = LlamaMLP.from_config(config.hidden_size, config.intermediate_size, jnp.float32)
    attention = LlamaAttention.from_config(config)
    attention = pz.nn.initialize_parameters(attention, jax.random.key(0))
    attention = pz.de.WithSideInputsFromInputTuple.handling(attention, tags=["attn_mask", "token_positions"])
    pz.ts.display(attention)
    result = attention((pz.nx.ones({"batch": 1, "seq": 4, "embedding": config.hidden_size}),
                        pz.nx.ones({"batch": 1, "seq": 4}),
                        pz.nx.arange("seq", 4)))
    return result


if __name__ == "__main__":
    main()
