# pretty much copied from https://github.com/google-deepmind/penzai/blob/main/penzai/example_models/gemma/model_core.py
import dataclasses
import itertools
import os
from argparse import Namespace
from typing import Iterable, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.sharding as jshard
import numpy as np
from penzai import pz  # ez
from penzai.toolshed import sharding_util

from .gguf import read_gguf
from .quantizers import make_linear, make_param
from .sharding import ConstrainedSharding, WithConstantSideInputsNonPytree


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
    act_fn: Literal["gelu", "silu"] = "silu"
    resid_rescale: float = 1.0
    attn_soft_cap: Optional[float] = None
    final_soft_cap: Optional[float] = None
    attn_scale_dim: Optional[int] = None
    pre_post_ln: bool = False


@pz.pytree_dataclass
class SoftCap(pz.Layer):
    soft_cap: float

    @pz.checked_layer_call
    def __call__(self, x: pz.nx.NamedArray) -> pz.nx.NamedArray:
        return self.soft_cap * jnp.tanh(x / self.soft_cap)

    def input_structure(self) -> pz.chk.StructureAnnotation:
        return pz.chk.Wildcard("x")

    def output_structure(self) -> pz.chk.StructureAnnotation:
        return pz.chk.Wildcard("x")


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class LlamaMLP(pz.nn.Sequential):
    @classmethod
    def from_config(cls, cfg: LlamaConfig):
        return cls([
            pz.nn.BranchAndMultiplyTogether(
                    branches=[
                    pz.nn.NamedGroup(
                        "gate",
                        [
                            pz.nn.add_parameter_prefix(
                                "gate_proj",
                                pz.nn.Linear.from_config(
                                    input_axes={"embedding": cfg.hidden_size},
                                    output_axes={"neurons": cfg.intermediate_size},
                                    dtype=cfg.parameter_dtype,
                                ),
                            ),
                            pz.nn.Elementwise(dict(
                                silu=jax.nn.silu,
                                gelu=jax.nn.gelu,
                            )[cfg.act_fn]),
                        ],
                    ),
                    pz.nn.add_parameter_prefix(
                        "up_proj",
                        pz.nn.Linear.from_config(
                            input_axes={"embedding": cfg.hidden_size},
                            output_axes={"neurons": cfg.intermediate_size},
                            dtype=cfg.parameter_dtype,
                        ),
                    )
                ]
            ),
            pz.nn.add_parameter_prefix(
                "out_proj",
                pz.nn.Linear.from_config(
                    input_axes={"neurons": cfg.intermediate_size},
                    output_axes={"embedding": cfg.hidden_size},
                    dtype=cfg.parameter_dtype,
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
    attn_scale_dim = projection_dim if config.attn_scale_dim is None else config.attn_scale_dim

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
                positions_tag="positions",
                embedding_axis="projection",
            ),
            pz.nn.ConstantRescale(
                by=jnp.array(
                    attn_scale_dim**-0.5, dtype=config.activation_dtype
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
                positions_tag="positions",
                embedding_axis="projection",
            ),
            pz.nn.CastToDType(config.activation_dtype),
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
            pz.nn.CastToDType(config.activation_dtype),
        ]),
        query_key_to_attn=pz.nn.Sequential([
            pz.nn.NamedEinsum(
                (
                    {"seq": "tq", "kv_heads": "h", "q_rep": "r", "projection": "p"},
                    {"seq": "tkv", "kv_heads": "h", "projection": "p"},
                ),
                {"seq": "tq", "kv_heads": "h", "q_rep": "r", "kv_seq": "tkv"},
            ),
        ] + ([SoftCap(config.attn_soft_cap)] if config.attn_soft_cap is not None else []) + [
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
            ConstrainedSharding.from_config(),
            pz.nn.Residual(pz.nn.Sequential([
                pz.nn.add_parameter_prefix(
                    "pre_attn_norm",
                    pz.nn.RMSLayerNorm.from_config(
                        across_axes={"embedding": config.hidden_size},
                        dtype=config.parameter_dtype,
                    ),
                ),
                pz.nn.CastToDType(config.activation_dtype),
                pz.nn.add_parameter_prefix(
                    "attn", LlamaAttention.from_config(config),
                ),
            ] + ([pz.nn.add_parameter_prefix(
                "post_attn_norm",
                pz.nn.RMSLayerNorm.from_config(
                    across_axes={"embedding": config.hidden_size},
                    dtype=config.parameter_dtype,
                ),
            )] if config.pre_post_ln else []))),
            ConstrainedSharding.from_config(),
            pz.nn.Residual(pz.nn.Sequential([
                pz.nn.add_parameter_prefix(
                    "pre_mlp_norm",
                    pz.nn.RMSLayerNorm.from_config(
                        across_axes={"embedding": config.hidden_size},
                        dtype=config.parameter_dtype,
                    ),
                ),
                pz.nn.CastToDType(config.activation_dtype),
                pz.nn.add_parameter_prefix(
                    "mlp", LlamaMLP.from_config(config),
                ),
            ] + ([pz.nn.add_parameter_prefix(
                "post_mlp_norm",
                pz.nn.RMSLayerNorm.from_config(
                    across_axes={"embedding": config.hidden_size},
                    dtype=config.parameter_dtype,
                ),
            )] if config.pre_post_ln else []))),
            ConstrainedSharding.from_config(),
        ])


@pz.pytree_dataclass
class LlamaInputs(pz.Struct):
    tokens: pz.nx.NamedArray
    attention_mask: pz.nx.NamedArray
    positions: pz.nx.NamedArray
    
    @classmethod
    def from_basic_segments(cls, tokens: pz.nx.NamedArray) -> "LlamaInputs":
        seq = tokens.named_shape["seq"]
        attention_mask = pz.nx.arange("seq", seq) >= pz.nx.arange("kv_seq", seq)
        return cls(
            tokens=tokens,
            attention_mask=attention_mask,
            positions=pz.nx.arange("seq", seq),
        )


@pz.pytree_dataclass
class LlamaTransformer(pz.Layer):
    config: LlamaConfig = dataclasses.field(metadata={"pytree_node": False})
    body: pz.LayerLike
    mesh: Optional[jshard.Mesh] = dataclasses.field(metadata={"pytree_node": False}, default=None)
    
    @pz.checked_layer_call
    def __call__(self, inputs: LlamaInputs) -> pz.nx.NamedArray:
        return self.body((inputs.tokens, inputs.positions, inputs.attention_mask))

    def input_structure(self) -> pz.chk.StructureAnnotation:
        return LlamaInputs(
            tokens=pz.chk.Wildcard("tokens"),
            positions=pz.chk.Wildcard("positions"),
            attention_mask=pz.chk.Wildcard("attention mask"),
        )

    def output_structure(self) -> pz.chk.StructureAnnotation:
        return pz.chk.Wildcard("unnormalized logits")
    
    @classmethod
    def from_config(cls, config: LlamaConfig, mesh: Optional[jshard.Mesh] = None) -> "LlamaTransformer":
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
                    pz.nn.ConstantRescale(
                        by=config.resid_rescale
                    ),
                    ConstrainedSharding.from_config(),
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
                ] + ([pz.nn.SoftCap(config.final_soft_cap)] if config.final_soft_cap is not None else []),
            ),
            tags=["positions", "attn_mask"],
        ),
        mesh=mesh,
    )

    @property
    def axis_name_to_mesh_name(self):
        base = {
            "neurons": "mp",
            "kv_heads": "mp",
            "seq": "sp",
            "batch": "dp",
        }
        return {**base, **{v: v for v in set(base.values())}}

    @property
    def inputs(self):
        return LlamaInputs

    @classmethod
    def make_mesh(cls, device_map: str):
        if device_map.startswith("auto"):
            _, *parts = device_map.split(":")
            mp = 1
            for part in parts:
                if part.startswith("mp="):
                    mp = int(part.partition("=")[2])
                # TODO SP support
            mesh = jshard.Mesh(np.asarray(jax.devices()).reshape((-1, 1, mp)), axis_names=("dp", "sp", "mp"))
        elif device_map.startswith("tpu:"):
            tpu_index = int(device_map.partition(":")[2])
            mesh = jshard.Mesh(np.asarray(jax.devices())[tpu_index:tpu_index+1].reshape((1, 1, 1)), axis_names=("dp", "sp", "mp"))
        else:
            raise ValueError(f"Unknown device map {device_map}")
        return mesh

    @classmethod
    def from_pretrained(cls, gguf_path: os.PathLike | Iterable[os.PathLike],
                        from_type: Literal[None, "gemma"] = None,
                        device_map="auto", extract_layer=None,
                        load_eager=False,
                        transpose_rotary: Optional[bool] = None,
                        load_on_cpu=False,
                        ):
        mesh = cls.make_mesh(device_map)
        
        gguf = read_gguf(gguf_path)
        is_gemma = from_type.startswith("gemma")
        if is_gemma:
            gguf.replace_metadata_prefix(f"{from_type}.", "llama.")
        config = LlamaConfig(
            vocab_size=gguf.metadata.get("llama.vocab_size", {None: 32_000, "gemma": 256_000, "gemma2": 256_000}[from_type]),
            hidden_size=gguf.metadata["llama.embedding_length"],
            intermediate_size=gguf.metadata["llama.feed_forward_length"],
            num_layers=gguf.metadata["llama.block_count"],
            num_attention_heads=gguf.metadata["llama.attention.head_count"],
            num_key_value_heads=gguf.metadata["llama.attention.head_count_kv"],
            act_fn={None: "silu", "gemma": "gelu", "gemma2": "gelu"}[from_type],
        )
        config.parameter_dtype = jnp.bfloat16
        config.activation_dtype = jnp.bfloat16  # 🤡
        if is_gemma:
            config.resid_rescale = jnp.sqrt(config.hidden_size).astype(config.activation_dtype)
        if "llama.attention.key_length" in gguf.metadata and "llama.attention.value_length" in gguf.metadata:
            assert gguf.metadata["llama.attention.key_length"] == gguf.metadata["llama.attention.value_length"], "Different key and value lengths not supported"
        scale_dim = gguf.metadata["llama.embedding_length"] // gguf.metadata["llama.attention.head_count"]
        if from_type == "gemma2":
            config.attn_scale_dim = scale_dim
            config.attn_soft_cap = gguf.metadata.get("attn_logit_softcapping")
            config.final_soft_cap = gguf.metadata.get("final_logit_softcapping")
        config.head_dim = gguf.metadata.get("llama.attention.key_length", scale_dim)

        transformer = cls.from_config(config, mesh=mesh)
        transformer = transformer.handle_sharding()
        
        if extract_layer is not None:
            assert isinstance(extract_layer, int)
            transformer = transformer.select().at_instances_of(LlamaBlock).apply_with_selected_index(
                lambda i, x: x if i < extract_layer else pz.nn.Identity()
            )
            transformer = transformer.select().at_instances_of(pz.nn.EmbeddingDecode).apply(lambda _: pz.nn.Identity())

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
            "unembed.embeddings": {None: "output.weight", "gemma": "token_embd.weight", "gemma2": "token_embd.weight"}[from_type],
        }
        for i in range(config.num_layers):
            if f"blk.{i}.post_attention_norm.weight" in gguf.keys():
                param_mapping[f"blocks.{i}.post_attn_norm.scale.weights"] = f"blk.{i}.post_attention_norm.weight"
            if f"blk.{i}.post_ffw_norm.weight" in gguf.keys():
                param_mapping[f"blocks.{i}.post_mlp_norm.scale.weights"] = f"blk.{i}.post_ffw_norm.weight"
        missing_keys = set(param_mapping.values()) - set(gguf.keys())
        if missing_keys:
            raise ValueError(f"Missing keys: {missing_keys}")
        is_transposed = {k: False for k in param_mapping}

        if transpose_rotary is None:
            transpose_rotary = not is_gemma
        missing_keys = set(gguf.keys())
        def remove_missing_key(k):
            missing_keys.remove(k)
            return k
        if not load_eager:
            # assume no linears are transposed
            transformer = transformer.select().at_instances_of(pz.nn.Linear).apply(
                lambda linear: make_linear(linear, *gguf[remove_missing_key(param_mapping[
                    linear.select().at_instances_of(pz.nn.UninitializedParameter).pick_nth_selected(0).get().name
                    ])], mesh=mesh, axis_name_to_mesh_name=transformer.axis_name_to_mesh_name,
                                           transpose_rotary=transpose_rotary, load_on_cpu=load_on_cpu)
            )
        transformer = transformer.select().at_instances_of(pz.nn.UninitializedParameter).apply(
            lambda param: make_param(param, *gguf[remove_missing_key(param_mapping[param.name])],
                                     mesh=mesh, axis_name_to_mesh_name=transformer.axis_name_to_mesh_name,
                                     is_transposed=is_transposed[param.name],
                                     transpose_rotary=transpose_rotary, load_on_cpu=load_on_cpu)
        )
        if missing_keys:
            raise ValueError(f"Missing keys: {missing_keys}")

        return transformer

    def handle_sharding(self, mod=ConstrainedSharding):
        return self.select().at_instances_of(mod).apply(
            lambda cs: WithConstantSideInputsNonPytree.handling(
                cs,
                {"axis_name_to_mesh_name": self.axis_name_to_mesh_name, "mesh": self.mesh}
            )
        )

    def to_tpu(self):
        return jax.device_put(
            self, sharding_util.name_to_name_sharding(
                self, self.mesh, self.axis_name_to_mesh_name, ignore_unnamed_arrays=True)
        )


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
