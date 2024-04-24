from penzai import pz
from typing import Literal, Tuple
from collections import OrderedDict
import jax.numpy as jnp
import numpy as np


def make_param(uninitialized_param: pz.nn.UninitializedParameter,
               quant_type: Literal["fp32", "int8", "int6", "int4"],
               tensor_data: Tuple[np.array],
               shape: Tuple[int]) -> pz.nn.Parameter:
    name = uninitialized_param.name 
    named_shape = uninitialized_param.value_structure.named_shape
    dtype = uninitialized_param.value_structure.dtype
    if quant_type == "fp32":
        dequantized = tensor_data[0]
    elif quant_type == "int8":
        dequantized = tensor_data[0].astype(dtype) * tensor_data[1]
    else:
        raise NotImplementedError(f"Quantization type {quant_type} not implemented")

    assert np.prod(shape) == dequantized.size
    assert np.prod(shape) == np.prod(list(named_shape.values()))
    print("pre", name, named_shape, shape, dequantized.shape)
    if name.endswith(".embeddings"):
        dequantized = dequantized.reshape(shape[::-1])
    else:
        dequantized = dequantized.reshape(shape[::-1])
        if ".attn.query" in name or ".attn.key" in name:
            # llama.cpp does rotary differently, i think
            head_dim = named_shape["projection"]
            dequantized = dequantized \
                .reshape(-1, head_dim // 2, 2, dequantized.shape[-1]) \
                    .swapaxes(1, 2) \
                        .reshape(dequantized.shape)  # taking the mayakovsky pill
        dequantized = dequantized.T  # for jax
    print("post", name, named_shape, shape, dequantized.shape)
    dequantized = jnp.asarray(dequantized.astype(dtype)).reshape(named_shape.values())
    dequantized = pz.nx.NamedArray(OrderedDict(named_shape), dequantized)
    # TODO make a custom ParameterLike for quantized parameters
    return pz.nn.Parameter(
        dequantized,
        name,
    )
