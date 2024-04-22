import jax.numpy as jnp
from penzai import pz  # ez
import dataclasses
import jax


@dataclasses.dataclass
class LLaMAConfig:
    vocab_size: int = 32_000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    qk_dim: int = 128
    v_dim: int = 128
    num_layers: int = 32


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class MLP(pz.nn.Sequential):
    @classmethod
    def from_config(cls, hidden_size: int, intermediate_size: int, dtype: jax.typing.DTypeLike):
        return cls([
            pz.nn.BranchAndMultiplyTogether(
                    branches=[
                    pz.nn.NamedGroup(
                        "gate",
                        [
                            pz.nn.add_parameter_prefix(
                                "gate_projection",
                                pz.nn.Linear.from_config(
                                    input_axes={"hidden": hidden_size},
                                    output_axes={"intermediate": intermediate_size},
                                    dtype=dtype,
                                ),
                            ),
                            pz.nn.Elementwise(jax.nn.silu),
                        ],
                    ),
                    pz.nn.add_parameter_prefix(
                        "up_projection",
                        pz.nn.Linear.from_config(
                            input_axes={"hidden": hidden_size},
                            output_axes={"intermediate": intermediate_size},
                            dtype=dtype,
                        ),
                    )
                ]
            ),
            pz.nn.add_parameter_prefix(
                "out_projection",
                pz.nn.Linear.from_config(
                    input_axes={"intermediate": intermediate_size},
                    output_axes={"hidden": hidden_size},
                    dtype=dtype,
                ),
            ),
        ])


...     
...
...


if __name__ == "__main__":
    config = LLaMAConfig()
    mlp = MLP.from_config(config.hidden_size, config.intermediate_size, jnp.float32)
