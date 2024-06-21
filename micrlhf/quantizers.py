import dataclasses
import warnings
from collections import OrderedDict
from functools import partial
from typing import Dict, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.sharding as jshard
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P
from penzai import pz
from penzai.toolshed import sharding_util


def make_linear(old_linear: pz.nn.Linear,
                quant_type: Literal["fp32", "q8_0", "q4_k", "q6_k"],
                tensor_data: Tuple[np.array],
                shape: Tuple[int],
                mesh: Optional[jshard.Mesh] = None,
                axis_name_to_mesh_name: Optional[Dict[str, str]] = None,
                load_on_cpu: bool = False,
                **kwargs
                ) -> pz.nn.Linear:
    param = old_linear.select().at_instances_of(pz.nn.UninitializedParameter).get()
    tensor_data, do_transpose = make_param(
        param, quant_type, tensor_data, shape, mesh, axis_name_to_mesh_name, return_metadata=True, **kwargs)
    if not do_transpose or quant_type == "fp32" or quant_type == "fp16":
        # fall back on dequantizing the parameter on HBM
        param = make_param(param, quant_type, tensor_data, shape, mesh, axis_name_to_mesh_name)
        return old_linear.select().at_instances_of(pz.nn.Parameter).apply(lambda p: param)
    new_data = []
    for d in tensor_data:
        d = d.reshape(*old_linear.output_axes.values(),
                        *list(old_linear.input_axes.values())[:-1], -1, d.shape[-1])
        d = device_put_named_sharded(d, (*old_linear.output_axes.keys(), *old_linear.input_axes.keys(), "quant_group"),
                                     mesh, axis_name_to_mesh_name, load_on_cpu=load_on_cpu)
        axes = (*old_linear.input_axes.keys(), "quant_group", *old_linear.output_axes.keys())
        d = d.untag(*axes).tag(*axes)
        new_data.append(d)
    def find_axis(keys):
        axes = [axis_name_to_mesh_name.get(k) for k in keys]
        if all(x is None for x in axes):
            return None
        return next(x for x in axes if x is not None)
    in_axis = find_axis(old_linear.input_axes.keys())
    out_axis = find_axis(old_linear.output_axes.keys())
    if quant_type == "q8_0":
        return Linear8bitTranspose(
            in_features=old_linear.input_axes,
            out_features=old_linear.output_axes,
            scale=new_data[0],
            quants=new_data[1],
            dtype=param.value_structure.dtype,
            mesh=mesh,
            batch_axis="dp",
            in_axis=in_axis,
            out_axis=out_axis
        ).speedup_matmul()
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


def matmul_8bit_kernel(inputs_ref, quants_ref, scale_ref, outputs_ref, accum_ref, *, block_k, quant_group_size=32):
    block_m = quants_ref.shape[2]
    
    accum_ref[...] = jnp.zeros_like(accum_ref)
    
    block_group = block_k // quant_group_size
    loop_iterations = max(1, inputs_ref.shape[-1] // block_k)
    def matmul_loop(i, _):
        quants = pl.load(quants_ref,
                            (pl.dslice(i * block_group, block_group),
                            slice(None), slice(None)))
        scale = pl.load(scale_ref,
                        (pl.dslice(i * block_group, block_group),
                        slice(None), slice(None)))
        scale = jnp.broadcast_to(scale.astype(jnp.float32), quants.shape)
        scaled = scale.reshape(block_k, block_m) * quants.reshape(block_k, block_m)
        inputs = pl.load(inputs_ref, (slice(None), pl.dslice(i*block_k, block_k)))
        result = jax.lax.dot_general(inputs.astype(jnp.bfloat16), scaled.astype(jnp.bfloat16),
                                    dimension_numbers=(((1,), (0,)), ((), ())),
                                    preferred_element_type=jnp.float32)
        accum_ref[...] += result
    jax.lax.fori_loop(0, loop_iterations, matmul_loop, init_val=None)
    outputs_ref[...] = accum_ref[...].astype(outputs_ref.dtype)


def matmul_fast(inputs, *tensors, kernel, mesh, batch_axis="dp", in_axis=None, out_axis=None):
    inputs = inputs.astype(jnp.bfloat16)
    tensors = [t if t.dtype.kind not in ("V", "f") else t.astype(jnp.bfloat16) for t in tensors]

    block_x, block_y, block_k = 256, 256, 512
    y = tensors[0].shape[2]
    batch_mesh = mesh.shape[batch_axis]
    per_block_size = inputs.shape[0] // batch_mesh
    input_size = inputs.shape[1]
    per_mp_input_size = input_size // (mesh.shape[in_axis] if in_axis is not None else 1)
    out_mesh = (mesh.shape[out_axis] if out_axis is not None else 1)
    per_mp_output_size = y // out_mesh
    if per_block_size < block_x:
        block_x = max(16, int(2 ** np.floor(np.log2(per_block_size))))
    if per_mp_input_size < block_k:
        block_k = max(16, int(2 ** np.floor(np.log2(per_mp_input_size))))
    if per_mp_output_size < block_y:
        block_y = max(16, int(2 ** np.floor(np.log2(per_mp_output_size))))
    x_pad = (block_x - per_block_size) % block_x
    k_pad = (block_k - per_mp_input_size) % block_k
    if x_pad or k_pad:
        inputs = jnp.pad(inputs.reshape(inputs.shape[0] // per_block_size, per_block_size, -1, per_mp_input_size),
                        ((0, 0), (0, x_pad),
                        (0, 0), (0, k_pad))
                        )
        inputs = inputs.reshape(-1, *inputs.shape[-2:])
        inputs = inputs.reshape(inputs.shape[0], -1)
    y_pad = (block_y - y) % block_y
    if y_pad:
        tensors = [jnp.pad(t, ((0, 0), (0, 0), (0, y_pad))) for t in tensors]

    def kernel_call(inputs, *tensors):
        outputs = pl.pallas_call(
            partial(kernel, block_k=block_k),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(int(inputs.shape[0] / block_x), int(per_mp_output_size / block_y)),
                in_specs=[
                    pl.BlockSpec(lambda i, j: (i, 0), (block_x, inputs.shape[1])),
                ] + [
                    pl.BlockSpec(lambda i, j: (0, 0, j), (t.shape[0], t.shape[1], block_y))
                    for t in tensors
                ],
                out_specs=pl.BlockSpec(
                    lambda i, j: (i, j), (block_x, block_y)
                ),
                scratch_shapes=[pltpu.VMEM((block_x, block_y), jnp.float32)],
            ),
            out_shape=jax.ShapeDtypeStruct((inputs.shape[0], per_mp_output_size), inputs.dtype),
            compiler_params=dict(mosaic=dict(dimension_semantics=("parallel", "arbitrary"))),
            interpret=True
        )(inputs, *tensors)
        if in_axis is not None:
            outputs = jax.lax.psum(outputs, axis_name=in_axis)
        return outputs

    result = jax.experimental.shard_map.shard_map(
        kernel_call,
        mesh=mesh,
        in_specs=(
            P(batch_axis, in_axis),
        ) + (
            P(in_axis, None, out_axis),
        ) * len(tensors),
        out_specs=P(batch_axis, out_axis),
        check_rep=False
    )(inputs, *tensors)
    if x_pad or y_pad:
        result = result.reshape(batch_mesh, inputs.shape[0] // batch_mesh, out_mesh, -1
                            )[:, :per_block_size, :, :per_mp_output_size]
        result = result.reshape(-1, *result.shape[-2:])
        result = result.reshape(result.shape[0], -1)
    return result


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class Linear8bitTranspose(QuantizedLinear):
    scale: pz.nx.NamedArray
    quants: pz.nx.NamedArray
    dtype: jax.typing.DTypeLike = dataclasses.field(metadata={"pytree_node": False})
    mesh: jshard.Mesh = dataclasses.field(metadata={"pytree_node": False})
    batch_axis: str = dataclasses.field(metadata={"pytree_node": False})
    in_axis: str = dataclasses.field(metadata={"pytree_node": False})
    out_axis: str = dataclasses.field(metadata={"pytree_node": False})
    sped_up: bool = dataclasses.field(metadata={"pytree_node": False}, default=False)

    @property
    def axes(self):
        return self.batch_axis, self.in_axis or "in_features", self.out_axis or "out_features"

    @property
    @jax.jit
    def sped_up_params(self):
        scale, quants = self.scale, self.quants
        if self.sped_up:
            return scale, quants
        _, ia, oa = self.axes
        scale, quants = (
            pz.nx.wrap(pz.nx.nmap(jnp.ravel)(
                pz.nx.nmap(jnp.ravel)(tensor.untag(*self.in_features.keys())).tag("in_features")
            .untag(*self.out_features.keys())).tag("out_features")
            .unwrap("in_features", "quant_group", "out_features"),
            ia, "quant_group", oa)
            for tensor in (scale, quants)
        )
        return scale, quants

    def speedup_matmul(self):
        scale, quants = self.sped_up_params
        return dataclasses.replace(self, scale=scale, quants=quants, sped_up=True)

    def quant_linear(self, inputs: jnp.ndarray) -> jnp.ndarray:
        _, ia, oa = self.axes
        scale, quants = self.sped_up_params
        scale, quants = (t.unwrap(ia, "quant_group", oa) for t in (scale, quants))
        mesh = self.mesh
        return matmul_fast(inputs, quants, scale, mesh=mesh, kernel=matmul_8bit_kernel,
                           batch_axis=self.batch_axis, in_axis=self.in_axis, out_axis=self.out_axis)
        # warnings.warn("Using slow 8-bit matmul, inputs too small")
        # weight = scale.astype(self.dtype) * quants
        # weight = weight.reshape(np.prod(list(self.in_features.values())), -1)
        # return inputs @ weight


def make_param(uninitialized_param: pz.nn.UninitializedParameter,
               quant_type: Literal["fp32", "q8_0", "q4_k", "q6_k"],
               tensor_data: Tuple[np.array],
               shape: Tuple[int],
               mesh: Optional[jshard.Mesh] = None,
               axis_name_to_mesh_name: Optional[Dict[str, str]] = None,
               return_metadata: bool = False,
               is_transposed: bool = False,
               transpose_rotary: bool = True,
               load_on_cpu: bool = False,
               ) -> pz.nn.Parameter:
    name = uninitialized_param.name 
    named_shape = uninitialized_param.value_structure.named_shape
    dtype = uninitialized_param.value_structure.dtype

    assert np.prod(shape) == np.prod(list(named_shape.values()))
    
    if (".attn.query" in name or ".attn.key" in name) and transpose_rotary:
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
    elif quant_type == "fp16":
        dequantized = tensor_data[0]
    elif quant_type == "q8_0":
        dequantized = tensor_data[0].astype(dtype) * tensor_data[1]
    else:
        raise NotImplementedError(f"Quantization type {quant_type} not implemented")

    dequantized = dequantized.reshape(shape[::-1])
    if do_transpose:
        dequantized = dequantized.T  # for jax
    if is_transposed:
        dequantized = dequantized.T
    dequantized = dequantized.astype(dtype).reshape(*named_shape.values())

    dequantized = device_put_named_sharded(dequantized, named_shape.keys(),
                                           mesh, axis_name_to_mesh_name,
                                           load_on_cpu=load_on_cpu)
    return pz.nn.Parameter(
        dequantized,
        name,
    )


def device_put_named_sharded(array, axis_names, mesh: Optional[jshard.Mesh], axis_name_to_mesh_name: Optional[Dict[str, str]],
                             load_on_cpu: bool = False):
    array = jax.device_put(array, jax.devices("cpu")[0])
    array = pz.nx.wrap(array, *axis_names)
    if load_on_cpu:
        return array
    if mesh is None or axis_name_to_mesh_name is None:
        return array
    return sharding_util.name_to_name_device_put(
        array,
        mesh,
        axis_name_to_mesh_name=axis_name_to_mesh_name,
    )

