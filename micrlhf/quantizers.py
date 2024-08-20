import dataclasses
import warnings
from collections import OrderedDict
from functools import partial
from typing import Callable, Dict, Literal, Optional, Tuple

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
    if not do_transpose or quant_type == "fp32" or quant_type == "fp16" or quant_type == "q6_k":
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
        return LinearQuantizedTranspose(
            params=new_data,
            in_features=old_linear.input_axes,
            out_features=old_linear.output_axes,
            dtype=param.value_structure.dtype,
            mesh=mesh,
            batch_axis="dp",
            in_axis=in_axis,
            out_axis=out_axis,
            kernel=matmul_8bit_kernel,
        ).speedup_matmul()
    elif quant_type == "q4_k":
        return LinearQuantizedTranspose(
            params=new_data,
            in_features=old_linear.input_axes,
            out_features=old_linear.output_axes,
            dtype=param.value_structure.dtype,
            mesh=mesh,
            batch_axis="dp",
            in_axis=in_axis,
            out_axis=out_axis,
            kernel=matmul_4bit_kernel,
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
        scale = scale.astype(jnp.float32)
        scaled = (scale * quants).reshape(block_k, block_m)
        inputs = pl.load(inputs_ref, (slice(None), pl.dslice(i*block_k, block_k)))
        result = jax.lax.dot_general(inputs.astype(jnp.bfloat16), scaled.astype(jnp.bfloat16),
                                    dimension_numbers=(((1,), (0,)), ((), ())),
                                    preferred_element_type=jnp.float32)
        accum_ref[...] += result
    jax.lax.fori_loop(0, loop_iterations, matmul_loop, init_val=None)
    outputs_ref[...] = accum_ref[...].astype(outputs_ref.dtype)


# let's ignore subnormals
def conv_f32(x):
    sr = jax.lax.shift_right_logical
    x = x.astype(jnp.int32)
    # x = jnp.where(x < 0, x + 65536, x)
    sign = sr(x, 31)
    exponent = sr(x, 10) & 0b11111

    fraction = x & 0x3ff

    sign_f32 = 1 - sign.astype(jnp.float32) * 2
    fraction_f32 = 1 + ((fraction.astype(jnp.float32)) / (1 << 10))
    exponent_f32 = jnp.exp2(exponent.astype(jnp.float32) - 15)
    return sign_f32 * fraction_f32 * exponent_f32


def matmul_4bit_kernel(inputs_ref,

                       scale_factors_ref, scale_offsets_ref,
                       qs1_ref, qs2_ref,

                       outputs_ref, accum_ref, *,

                       block_k, quant_group_size=256,
                       ):
    sr = lambda x, s: jax.lax.shift_right_logical(x, s)

    assert quant_group_size == 256

    block_step = block_k // 256
    num_blocks = scale_factors_ref.shape[1] // block_step
    assert scale_factors_ref.shape[1] == scale_offsets_ref.shape[1] == qs1_ref.shape[1] == qs2_ref.shape[1]
    assert num_blocks >= 0
    assert block_k == block_step * 256

    loop_iterations = num_blocks
    accum_ref[...] = jnp.zeros_like(accum_ref)

    block_m = qs2_ref.shape[-1]
    assert block_m == scale_factors_ref.shape[-1] == scale_offsets_ref.shape[-1] == qs1_ref.shape[-1]
    assert block_k == (inputs_ref.shape[1] // loop_iterations)
    block_n = inputs_ref.shape[0]
    def matmul_loop(i, _):
        block_slice = pl.dslice(i * block_step, block_step)
        a = slice(None)
        qs1 = pl.load(qs1_ref, (a, block_slice, a))
        qs2 = pl.load(qs2_ref, (a, block_slice, a))

        qs1 = qs1.astype(jnp.int32)
        qs2 = qs2.astype(jnp.int32)
        i8tou8 = lambda x: jnp.where(x < 0, 256 + x, x)
        qs1 = i8tou8(qs1)
        qs2 = i8tou8(qs2)

        scale_factors = conv_f32(pl.load(scale_factors_ref, (a, block_slice, a)))
        scale_offsets = conv_f32(pl.load(scale_offsets_ref, (a, block_slice, a)))

        qs1 = qs1.reshape(12, 1, *qs1.shape[1:])
        qs2 = qs2.reshape(4, 32, *qs2.shape[1:])

        scale_factors = scale_factors.reshape(1, 1, *scale_factors.shape[1:])
        scale_offsets = scale_offsets.reshape(1, 1, *scale_offsets.shape[1:])

        inputs = pl.load(inputs_ref, (a, pl.dslice(i*block_k, block_k))).astype(jnp.float32)

        chunk1 = qs1[0:4]
        chunk2 = qs1[4:8]
        chunk3 = qs1[8:]
        factor_scale = jnp.concatenate([chunk1 & 0b111111, (chunk3 & 15) | (sr(chunk1, 6) << 4)], axis=0)
        offset_scale = jnp.concatenate([chunk2 & 0b111111, (sr(chunk3, 4) & 0b1111) | (sr(chunk2, 6) << 4)], axis=0)

        # basify = lambda x: x  # x.astype(jnp.int8) if not based else x
        basify = lambda x: x.astype(jnp.float32)
        factors = scale_factors * basify(factor_scale)
        offsets = scale_offsets * basify(offset_scale)

        # max 15
        # print(qs2.shape)
        # qs2 = jnp.stack([qs2 & 0xf, sr(qs2, 4)], axis=1).reshape(8, 32, num_blocks, -1)
        qs2 = jnp.concatenate([qs2 & 0xf, sr(qs2, 4)], axis=1).reshape(8, 32, block_step, -1)

        matrix = factors * basify(qs2) - offsets
        # matrix = basify(qs2)

        # slightly_less_of_an_abomination = jnp.concatenate([ab(x) for x in [qs2[0], qs2[1], qs2[2], qs2[3]]] * 2, axis=1)
        # matrix = slightly_less_of_an_abomination
        matrix = matrix.reshape(block_k, block_m)
        inputs = inputs.reshape(block_n, block_k).astype(jnp.float32)
        result = inputs @ matrix

        accum_ref[...] += result
    jax.lax.fori_loop(0, loop_iterations, matmul_loop, init_val=None, unroll=True)
    outputs_ref[...] = accum_ref[...].astype(outputs_ref.dtype)


def matmul_fast(inputs, *tensors, kernel, mesh, batch_axis="dp", in_axis=None, out_axis=None):
    is_transpose = False if kernel.__name__ not in ("matmul_4bit_kernel",) else True
    is_f16 = False if kernel.__name__ not in ("matmul_4bit_kernel",) else True

    inputs = inputs.astype(jnp.bfloat16)
    tensors = [t if t.dtype.kind not in ("V", "f") else (t.astype(jnp.bfloat16) if not is_f16 else t.view(jnp.int16)) for t in tensors]
    # tensors = [t.view(jnp.int8) if t.dtype == jnp.uint8 else t for t in tensors]

    block_x, block_y, block_k = 256, 256, 512
    if kernel.__name__ == "matmul_4bit_kernel":
        block_x, block_y, block_k = 128, 128, 256 * 8
    y = tensors[0].shape[2]
    batch_mesh = mesh.shape[batch_axis]
    per_block_size = inputs.shape[0] // batch_mesh
    input_size = inputs.shape[1]
    per_mp_input_size = input_size // (mesh.shape[in_axis] if in_axis is not None else 1)
    out_mesh = (mesh.shape[out_axis] if out_axis is not None else 1)
    per_mp_output_size = y // out_mesh
    if per_block_size < block_x:
        block_x = max(16, int(2 ** np.floor(np.log2(per_block_size))))
    if per_mp_input_size < block_k and kernel.__name__ != "matmul_4bit_kernel":
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
        if is_transpose:
            batch_dims = inputs.shape[:-1]
            inputs = inputs.reshape(*batch_dims, -1, block_k)
            inputs = inputs.swapaxes(-2, -1)
            inputs = inputs.reshape(*batch_dims, -1)

        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(int(inputs.shape[0] / block_x), int(per_mp_output_size / block_y)),
            in_specs=[
                pl.BlockSpec(lambda i, j: (i, 0), (block_x, inputs.shape[1])),
            ] + [
                pl.BlockSpec(lambda i, j: (0, 0, j),
                                (t.shape[0], t.shape[1], block_y))
                for t in tensors
            ]
            # + ([] if kernel.__name__ != "matmul_4bit_kernel" else [
            #     pl.BlockSpec(lambda i, j: (0, 0), tensors[-1].shape)
            # ])
            ,
            out_specs=pl.BlockSpec(
                lambda i, j: (i, j), (block_x, block_y)
            ),
            scratch_shapes=[pltpu.VMEM((block_x, block_y), jnp.float32)],
        )
        outputs = pl.pallas_call(
            partial(kernel, block_k=block_k),
            grid_spec=grid_spec,
            out_shape=jax.ShapeDtypeStruct((inputs.shape[0], per_mp_output_size), inputs.dtype),
            compiler_params=dict(mosaic=dict(dimension_semantics=("parallel", "parallel"))),
            # interpret=True
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
class LinearQuantizedTranspose(QuantizedLinear):
    kernel: Callable = dataclasses.field(metadata={"pytree_node": False})
    params: Tuple[pz.nx.NamedArray]
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
        if self.sped_up:
            return self.params
        _, ia, oa = self.axes
        params = self.params
        params = tuple(
            pz.nx.wrap(pz.nx.nmap(jnp.ravel)(
                pz.nx.nmap(jnp.ravel)(tensor.untag(*self.in_features.keys())).tag("in_features")
            .untag(*self.out_features.keys())).tag("out_features")
            .unwrap("in_features", "quant_group", "out_features"),
            ia, "quant_group", oa)
            for tensor in params
        )
        return params

    def speedup_matmul(self):
        return dataclasses.replace(self, params=self.sped_up_params, sped_up=True)

    def quant_linear(self, inputs: jnp.ndarray) -> jnp.ndarray:
        _, ia, oa = self.axes
        params = self.sped_up_params
        params = (t.unwrap(ia, "quant_group", oa) for t in params)
        mesh = self.mesh
        return matmul_fast(inputs, *params, mesh=mesh, kernel=self.kernel,
                           batch_axis=self.batch_axis, in_axis=self.in_axis, out_axis=self.out_axis)


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
    elif quant_type == "q4_k":
        scale_factors, scale_offsets, qs1, qs2 = tensor_data

        num_blocks = scale_factors.shape[0]
        assert num_blocks == scale_offsets.shape[0] == qs1.shape[0] == qs2.shape[0]

        scale_factors = scale_factors.reshape(num_blocks, 1, 1, -1)
        scale_offsets = scale_offsets.reshape(num_blocks, 1, 1, -1)
        qs1 = qs1.reshape(num_blocks, 12, 1, -1)
        qs2 = qs2.reshape(num_blocks, 4, 32, -1)

        factors = scale_factors * jnp.concatenate([qs1[:, 0:4] & 0b111111, (qs1[:, 8:] & 15) | ((qs1[:, 0:4] >> 6) << 4)], axis=1)
        offsets = scale_offsets * jnp.concatenate([qs1[:, 4:8] & 0b111111, (qs1[:, 8:] >> 4) | ((qs1[:, 4:8] >> 6) << 4)], axis=1)
        
        qs2 = jnp.stack([qs2 & 0xf, qs2 >> 4], axis=2).reshape(num_blocks, 8, 32, -1)

        dequantized = factors * qs2 - offsets
    elif quant_type == "q6_k":
        scales, ql, qh, sc = tensor_data
        
        scales = scales.reshape(*scales.shape, -1)
        ql = ql.reshape(*ql.shape, -1)
        qh = qh.reshape(*qh.shape, -1)
        sc = sc.reshape(*sc.shape, -1)
        
        num_blocks = scales.shape[0]
        assert num_blocks == ql.shape[0] == qh.shape[0] == sc.shape[0]

        # scales: nb, 1, x (float32)
        # ql: nb, 128, x (int16)
        # qh: nb, 64, x (int16)
        # sc: nb, 16, x (float32)
        
        sc = sc.reshape(num_blocks, 16, 1, -1)

        q1 = (ql[:,   :32 ] & 0xF) | (((qh[:, :32] >> 0) & 3) << 4) - 32
        q2 = (ql[:, 32:64 ] & 0xF) | (((qh[:, :32] >> 2) & 3) << 4) - 32
        q3 = (ql[:,   :32 ] >>  4) | (((qh[:, :32] >> 4) & 3) << 4) - 32
        q4 = (ql[:, 32:64 ] >>  4) | (((qh[:, :32] >> 6) & 3) << 4) - 32
        q5 = (ql[:, 64:96 ] & 0xF) | (((qh[:, 32:] >> 0) & 3) << 4) - 32
        q6 = (ql[:, 96:128] & 0xF) | (((qh[:, 32:] >> 2) & 3) << 4) - 32
        q7 = (ql[:, 64:96 ] >>  4) | (((qh[:, 32:] >> 4) & 3) << 4) - 32
        q8 = (ql[:, 96:128] >>  4) | (((qh[:, 32:] >> 6) & 3) << 4) - 32

        dequantized = scales * np.concatenate([
            sc[:,  0] * q1[:, :16],
            sc[:,  1] * q1[:, 16:],
            sc[:,  2] * q2[:, :16],
            sc[:,  3] * q2[:, 16:],
            sc[:,  4] * q3[:, :16],
            sc[:,  5] * q3[:, 16:],
            sc[:,  6] * q4[:, :16],
            sc[:,  7] * q4[:, 16:],
            sc[:,  8] * q5[:, :16],
            sc[:,  9] * q5[:, 16:],
            sc[:, 10] * q6[:, :16],
            sc[:, 11] * q6[:, 16:],
            sc[:, 12] * q7[:, :16],
            sc[:, 13] * q7[:, 16:],
            sc[:, 14] * q8[:, :16],
            sc[:, 15] * q8[:, 16:],
        ], axis=1)
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

