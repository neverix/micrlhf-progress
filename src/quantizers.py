from penzai import pz
from typing import Literal, Tuple
import jax.numpy as jnp
import numpy as np


def make_param(uninitialized_param: pz.nn.UninitializedParameter,
               quant_type: Literal["fp32", "int8", "int6", "int4"],
               tensor_data: Tuple[np.array],
               shape: Tuple[int]) -> pz.nn.Parameter:
    dtype = uninitialized_param.value_structure.dtype
    if quant_type == "fp32":
        dequantized = tensor_data[0]
    elif quant_type == "int8":
        dequantized = tensor_data[0].astype(dtype) * tensor_data[1]
    else:
        raise NotImplementedError(f"Quantization type {quant_type} not implemented")

    dequantized = jnp.asarray(dequantized.astype(dtype)).reshape(shape)
    dequantized = pz.nx.NamedArray(uninitialized_param.value_structure.named_shape.keys(), dequantized)
    return pz.nn.Parameter(
        dequantized,
        uninitialized_param.name,
    )
