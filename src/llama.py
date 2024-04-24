# pretty much copied from https://github.com/google-deepmind/penzai/blob/main/penzai/example_models/gemma/model_core.py
import dataclasses
import os

import jax
import jax.numpy as jnp
import jax.sharding as jshard
import numpy as np
from penzai import pz  # ez

from .gguf import GGUFReader
from .quantizers import make_param


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
    num_heads = num_kv_heads
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
                    jnp.finfo(config.activation_dtype).min, dtype=config.activation_dtype
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
                        "kv_heads": num_heads,
                        "q_rep": q_rep,
                        "projection": projection_dim,
                    },
                    output_axes={"embedding": hidden_size},
                    dtype=config.parameter_dtype,
                ),
            ),
        ]),
    )


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class LlamaBlock(pz.nn.Sequential):
    @classmethod
    def from_config(cls, config: LlamaConfig):
        return cls([
            pz.nn.Residual(pz.nn.Sequential([
                pz.nn.add_parameter_prefix(
                    "pre_attn_norm",
                    pz.nn.RMSLayerNorm.from_config(
                        across_axes={"embedding": config.hidden_size},
                        dtype=config.parameter_dtype,
                    ),
                ),
                pz.nn.add_parameter_prefix(
                    "attn", LlamaAttention.from_config(config),
                ),
            ])),
            pz.nn.Residual(pz.nn.Sequential([
                pz.nn.add_parameter_prefix(
                    "pre_mlp_norm",
                    pz.nn.RMSLayerNorm.from_config(
                        across_axes={"embedding": config.hidden_size},
                        dtype=config.parameter_dtype,
                    ),
                ),
                pz.nn.add_parameter_prefix(
                    "mlp", LlamaMLP.from_config(config.hidden_size, config.intermediate_size, config.parameter_dtype),
                ),
            ])),
        ])


@pz.pytree_dataclass
class LlamaInputs(pz.Struct):
    tokens: pz.nx.NamedArray
    attention_mask: pz.nx.NamedArray
    token_positions: pz.nx.NamedArray
    
    @classmethod
    def from_basic_segments(cls, tokens: pz.nx.NamedArray) -> "LlamaInputs":
        seq = tokens.named_shape["seq"]
        attention_mask = pz.nx.arange("seq", seq) >= pz.nx.arange("kv_seq", seq)
        return cls(
            tokens=tokens,
            attention_mask=attention_mask,
            token_positions=pz.nx.arange("seq", seq),
        )


@pz.pytree_dataclass
class LlamaTransformer(pz.Layer):
    config: LlamaConfig = dataclasses.field(metadata={"pytree_node": False})
    body: pz.LayerLike
    
    @pz.checked_layer_call
    def __call__(self, inputs: LlamaInputs) -> pz.nx.NamedArray:
        return self.body((inputs.tokens, inputs.attention_mask, inputs.token_positions))

    def input_structure(self) -> pz.chk.StructureAnnotation:
        return LlamaInputs(
            tokens=pz.chk.Wildcard("tokens"),
            token_positions=pz.chk.Wildcard("positions"),
            attention_mask=pz.chk.Wildcard("attention mask"),
        )

    def output_structure(self) -> pz.chk.StructureAnnotation:
        return pz.chk.Wildcard("unnormalized logits")
    
    @classmethod
    def from_config(cls, config: LlamaConfig) -> "LlamaTransformer":
        return cls(
            config=config,
            body=pz.de.WithSideInputsFromInputTuple.handling(
                pz.nn.Sequential([
                    pz.nn.add_parameter_prefix(
                        "embed",
                        pz.nn.EmbeddingLookup(
                            pz.nn.EmbeddingTable.from_config(
                                vocab_size=config.vocab_size,
                                embedding_axes={"embedding": config.hidden_size},
                                dtype=config.parameter_dtype,
                            )
                        )
                    ),
                    pz.nn.CastToDType(config.activation_dtype),
                    pz.nn.add_parameter_prefix(
                        "blocks",
                        pz.nn.Sequential([
                            pz.nn.add_parameter_prefix(
                                str(i), LlamaBlock.from_config(config)
                            ) for i in range(config.num_layers)
                        ]),
                    ),
                    pz.nn.add_parameter_prefix(
                        "final_norm",
                        pz.nn.RMSLayerNorm.from_config(
                            across_axes={"embedding": config.hidden_size},
                            dtype=config.parameter_dtype,
                        ),
                    ),
                    pz.nn.add_parameter_prefix(
                        "unembed",
                        pz.nn.EmbeddingDecode(
                            pz.nn.EmbeddingTable.from_config(
                                vocab_size=config.vocab_size,
                                embedding_axes={"embedding": config.hidden_size},
                                dtype=config.parameter_dtype,
                            )
                        )
                    )
                ]),
                tags=["attn_mask", "token_positions"],
            )
        )

    @property
    def axis_name_to_mesh_name(self):
        return {
            "neurons": "mp",
            "kv_heads": "mp"
        }

    @classmethod
    def from_pretrained(cls, gguf_path: os.PathLike, device_map="auto"):
        if device_map != "auto":
            raise ValueError(
                "I'm actually not sure yet how device_map will be handled, " \
                "this is just to mimic HF's API.")
        mesh = jshard.Mesh(np.asarray(jax.devices()).reshape((1, 1, -1)), axis_names=("dp", "sp", "tp"))
        
        gguf = GGUFReader(gguf_path)
        config = LlamaConfig(
            vocab_size=gguf.metadata["llama.vocab_size"],
            hidden_size=gguf.metadata["llama.embedding_length"],
            intermediate_size=gguf.metadata["llama.feed_forward_length"],
            num_layers=gguf.metadata["llama.block_count"],
            num_attention_heads=gguf.metadata["llama.attention.head_count"],
            num_key_value_heads=gguf.metadata["llama.attention.head_count_kv"],
        )
        config.head_dim = gguf.metadata["llama.embedding_length"] // gguf.metadata["llama.attention.head_count"]
        config.parameter_dtype = jnp.bfloat16
        config.activation_dtype = jnp.float16
        
        transformer = cls.from_config(config)

        param_mapping = {
            "embed.embeddings": "token_embd.weight",
            **{f"blocks.{i}.pre_attn_norm.scale.weights": f"blk.{i}.attn_norm.weight" for i in range(config.num_layers)},
            **{f"blocks.{i}.attn.query.weights": f"blk.{i}.attn_q.weight" for i in range(config.num_layers)},
            **{f"blocks.{i}.attn.key.weights": f"blk.{i}.attn_k.weight" for i in range(config.num_layers)},
            **{f"blocks.{i}.attn.value.weights": f"blk.{i}.attn_v.weight" for i in range(config.num_layers)},
            **{f"blocks.{i}.attn.output.weights": f"blk.{i}.attn_output.weight" for i in range(config.num_layers)},
            **{f"blocks.{i}.pre_mlp_norm.scale.weights": f"blk.{i}.ffn_norm.weight" for i in range(config.num_layers)},
            **{f"blocks.{i}.mlp.gate_proj.weights": f"blk.{i}.ffn_gate.weight" for i in range(config.num_layers)},
            **{f"blocks.{i}.mlp.up_proj.weights": f"blk.{i}.ffn_up.weight" for i in range(config.num_layers)},
            **{f"blocks.{i}.mlp.out_proj.weights": f"blk.{i}.ffn_down.weight" for i in range(config.num_layers)},
            "final_norm.scale.weights": "output_norm.weight",
            "unembed.embeddings": "output.weight",           
        }

        # transformer = transformer.select().at_instances_of(pz.nn.Linear).apply(
        #     lambda linear: make_linear(linear, *gguf[param_mapping[
        #         linear.select().at_instances_of(pz.nn.UninitializedParameter).pick_nth_selected(0).get().name
        #         ]])
        # )
        transformer = transformer.select().at_instances_of(pz.nn.UninitializedParameter).apply(
            lambda param: make_param(param, *gguf[param_mapping[param.name]],
                                     mesh=mesh, axis_name_to_mesh_name=transformer.axis_name_to_mesh_name)
        )
        
        return transformer


def main():
    transformer = LlamaTransformer.from_pretrained("models/Meta-Llama-3-8B-Instruct.Q8_0.gguf")
    transformer = (
        transformer.select()
        .at_instances_of(pz.nn.UninitializedParameter)
        .apply(lambda param: param.initialize_with_value(
            pz.nx.zeros(param.value_structure.named_shape, dtype=param.value_structure.dtype)
        ))
    )
    result = transformer(LlamaInputs.from_basic_segments(
        pz.nx.ones({"seq": 4}, dtype=jnp.int32)))
    return result


if __name__ == "__main__":
    main()
