import dataclasses
from collections import OrderedDict
from typing import Dict, Literal, Optional, Tuple

import jax
import warnings
from jax.experimental import pallas as pl
import jax.numpy as jnp
import jax.sharding as jshard
import numpy as np
from penzai import pz
from penzai.toolshed import sharding_util


def make_linear(old_linear: pz.nn.Linear,
                quant_type: Literal["fp32", "q8_0", "q4_k", "q6_k"],
                tensor_data: Tuple[np.array],
                shape: Tuple[int],
                mesh: Optional[jshard.Mesh] = None,
                axis_name_to_mesh_name: Optional[Dict[str, str]] = None,
                ) -> pz.nn.Linear:
    param = old_linear.select().at_instances_of(pz.nn.UninitializedParameter).get()
    tensor_data, do_transpose = make_param(
    param, quant_type, tensor_data, shape, mesh, axis_name_to_mesh_name, return_metadata=True)
    if not do_transpose or quant_type == "fp32":
        # fall back on dequantizing the parameter on HBM
        param = make_param(param, quant_type, tensor_data, shape, mesh, axis_name_to_mesh_name)
        return old_linear.select().at_instances_of(pz.nn.Parameter).apply(lambda p: param)
    new_data = []
    for d in tensor_data:
        d = d.reshape(*old_linear.output_axes.values(),
                        *list(old_linear.input_axes.values())[:-1], -1, d.shape[-1])
        d = jax.device_put(d)  # TODO replicate
        d = pz.nx.wrap(d, *old_linear.output_axes.keys(), *old_linear.input_axes.keys(), "quant_group")
        axes = (*old_linear.input_axes.keys(), "quant_group", *old_linear.output_axes.keys())
        d = d.untag(*axes).tag(*axes)
        new_data.append(d)
    if quant_type == "q8_0":
        return Linear8bitTranspose(
            in_features=old_linear.input_axes,
            out_features=old_linear.output_axes,
            scale=new_data[0],
            quants=new_data[1],
            dtype=param.value_structure.dtype
        )
    else:
        raise NotImplementedError(f"Quantization type {quant_type} not implemented")


@pz.pytree_dataclass
class QuantizedLinear(pz.Layer):
    in_features: OrderedDict[str, int] = dataclasses.field(metadata={"pytree_node": False})
    out_features: OrderedDict[str, int] = dataclasses.field(metadata={"pytree_node": False})

    @pz.checked_layer_call
    def __call__(self, inputs: pz.nx.NamedArray) -> pz.nx.NamedArray:
        orig_shape = inputs.named_shape
        batch_shape = OrderedDict((k, v) for k, v in orig_shape.items() if k not in self.in_features)
        inputs = pz.nx.nmap(jnp.ravel)(inputs.untag(*self.in_features.keys())).tag("in_features")
        inputs = pz.nx.nmap(jnp.ravel)(inputs.untag(*batch_shape.keys())).tag("batch")
        outputs = pz.nx.wrap(self.quant_linear(inputs.unwrap("batch", "in_features")), "batch", "out_features")
        outputs = pz.nx.nmap(lambda x: x.reshape(*batch_shape.values()))(outputs.untag("batch")).tag(*batch_shape.keys())
        outputs = pz.nx.nmap(lambda x: x.reshape(*self.out_features.values()))(outputs.untag("out_features")).tag(*self.out_features.keys())
        return outputs

    def input_structure(self) -> pz.chk.StructureAnnotation:
        return pz.chk.Wildcard("input features")

    def output_structure(self) -> pz.chk.StructureAnnotation:
        return pz.chk.Wildcard("output features")


def matmul_8bit_kernel(quants_ref, scale_ref, inputs_ref, outputs_ref):
    quants = quants_ref[...]
    scale = jnp.broadcast_to(scale_ref[...].astype(jnp.float32), quants.shape).astype(jnp.bfloat16)
    quants, scale = quants.reshape(-1, quants.shape[-1]), scale.reshape(-1, scale.shape[-1])
    scaled = jax.lax.mul(scale.astype(jnp.float32), quants.astype(jnp.float32))
    result = jax.lax.dot_general(inputs_ref[...], scaled, dimension_numbers=(((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    outputs_ref[...] = result.astype(outputs_ref.dtype)


def matmul_8bit_fast(quants, scale, inputs):
    inputs = inputs.astype(jnp.bfloat16)
    scale = scale.astype(jnp.bfloat16)

    block_x, block_y = 16, 128
    return pl.pallas_call(
        matmul_8bit_kernel,
        out_shape=jax.ShapeDtypeStruct((inputs.shape[0], quants.shape[2]), inputs.dtype),
        grid=(int(inputs.shape[0] / block_x), int(quants.shape[2] / block_y)),
        in_specs=[
            pl.BlockSpec(lambda i, j: (0, 0, j), (quants.shape[0], quants.shape[1], block_y)),
            pl.BlockSpec(lambda i, j: (0, 0, j), (quants.shape[0], 1, block_y)),
            pl.BlockSpec(lambda i, j: (i, 0), (block_x, inputs.shape[1])),
        ],
        out_specs=pl.BlockSpec(
            lambda i, j: (i, j), (block_x, block_y)
        ),
        compiler_params=dict(mosaic=dict(dimension_semantics=("parallel", "parallel"))),
    )(quants, scale, inputs)


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class Linear8bitTranspose(QuantizedLinear):
    scale: pz.nx.NamedArray
    quants: pz.nx.NamedArray
    dtype: jax.typing.DTypeLike = dataclasses.field(metadata={"pytree_node": False})

    def quant_linear(self, inputs: jnp.ndarray) -> jnp.ndarray:
        scale, quants = self.scale, self.quants
        scale, quants = (
            pz.nx.nmap(jnp.ravel)(
                pz.nx.nmap(jnp.ravel)(tensor.untag(*self.in_features.keys())).tag("in_features")
            .untag(*self.out_features.keys())).tag("out_features")
            .unwrap("in_features", "quant_group", "out_features")
            for tensor in (scale, quants)
        )
        if inputs.shape[0] >= 16:
            return matmul_8bit_fast(quants, scale, inputs)
        warnings.warn("Using slow 8-bit matmul, inputs too small")
        weight = scale.astype(self.dtype) * quants
        weight = weight.reshape(np.prod(list(self.in_features.values())), -1)
        return inputs @ weight


def make_param(uninitialized_param: pz.nn.UninitializedParameter,
               quant_type: Literal["fp32", "q8_0", "q4_k", "q6_k"],
               tensor_data: Tuple[np.array],
               shape: Tuple[int],
               mesh: Optional[jshard.Mesh] = None,
               axis_name_to_mesh_name: Optional[Dict[str, str]] = None,
               return_metadata: bool = False,
               ) -> pz.nn.Parameter:
    name = uninitialized_param.name 
    named_shape = uninitialized_param.value_structure.named_shape
    dtype = uninitialized_param.value_structure.dtype
    
    assert np.prod(shape) == np.prod(list(named_shape.values()))
    
    if ".attn.query" in name or ".attn.key" in name:
        new_data = []
        for d in tensor_data:
            # llama.cpp does rotary differently, i think
            head_dim = named_shape["projection"]
            embed_dim = named_shape["embedding"]
            n_heads = np.prod(list(named_shape.values())) // head_dim // embed_dim
            d = d \
                .reshape(n_heads, head_dim // 2, 2, -1, *d.shape[1:]) \
                    .swapaxes(1, 2) \
                        .reshape(d.shape)  # taking the mayakovsky pill
            new_data.append(d)
        tensor_data = new_data

    do_transpose = not name.endswith(".embeddings")
    if return_metadata:
        return tensor_data, do_transpose

    if quant_type == "fp32":
        dequantized = tensor_data[0]
    elif quant_type == "q8_0":
        dequantized = tensor_data[0].astype(dtype) * tensor_data[1]
    else:
        raise NotImplementedError(f"Quantization type {quant_type} not implemented")

    dequantized = dequantized.reshape(shape[::-1])
    if do_transpose:
        dequantized = dequantized.T  # for jax
    dequantized = dequantized.astype(dtype).reshape(*named_shape.values())
    # TODO replicate
    dequantized = jax.device_put(dequantized)
    dequantized = pz.nx.wrap(dequantized, *named_shape.keys())
    return pz.nn.Parameter(
        dequantized,
        name,
    )
