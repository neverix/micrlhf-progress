from micrlhf.quantizers import Linear8bitTranspose, matmul_8bit_fast
import jax.numpy as jnp
from penzai import pz
import numpy as np
import jax


def matmul_8bit(quants, scale, inputs):
    return inputs @ (quants * scale).reshape(inputs.shape[-1], -1)


def test_tpu_stuff_works():
    a = np.array([1, 2, 3]) + 1
    b = jnp.array([-4, -5, -6])
    b = jax.device_put(b, jax.devices("tpu")[0])
    c = np.asarray(a + b).tolist()
    assert c == [-2, -2, -2]

def test_make_mesh():
    mesh = jax.sharding.Mesh(np.asarray(jax.devices("tpu"))[:1].reshape(-1, 1), ("dp", "mp"))
    mesh = jax.sharding.Mesh(np.asarray(jax.devices("tpu")).reshape(-1, 2), ("dp", "mp"))

def test_simple_matmul():
    mesh = jax.sharding.Mesh(np.asarray(jax.devices("tpu"))[:1].reshape(-1, 1), ("dp", "mp"))

    a, b, c, bs = 512, 256, 256, 32
    quants = jax.random.randint(jax.random.PRNGKey(0), (a // bs, bs, b), 0, 255, dtype=jnp.int8)
    scale = jax.random.normal(jax.random.PRNGKey(1), (a // bs, 1, b), dtype=jnp.bfloat16) / 255
    inputs = jax.random.normal(jax.random.PRNGKey(2), (c, a), dtype=jnp.bfloat16)
    outputs = matmul_8bit(quants, scale, inputs)
    outputs_fast = matmul_8bit_fast(quants, scale, inputs, mesh)
    assert jnp.max(jnp.abs(outputs_fast - outputs)) == 0

def test_dp_matmul():
    mesh = jax.sharding.Mesh(np.asarray(jax.devices("tpu")).reshape(-1, 1), ("dp", "mp"))

    a, b, c, bs = 512, 256, 512, 32
    for order in (-1, 1):
        in_axis, out_axis = ("mp", None)[::order]
        quants = jax.random.randint(jax.random.PRNGKey(0), (a // bs, bs, b), 0, 255, dtype=jnp.int8)
        scale = jax.random.normal(jax.random.PRNGKey(1), (a // bs, 1, b), dtype=jnp.bfloat16) / 255
        inputs = jax.random.normal(jax.random.PRNGKey(2), (c, a), dtype=jnp.bfloat16)
        
        quants_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(in_axis, None, out_axis))
        scale_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(in_axis, None, out_axis))
        inputs_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp", in_axis))
        quants = jax.device_put(quants, quants_sharding)
        scale = jax.device_put(scale, scale_sharding)
        inputs = jax.device_put(inputs, inputs_sharding)

        outputs = matmul_8bit(quants, scale, inputs)
        outputs_fast = matmul_8bit_fast(quants, scale, inputs, mesh, in_axis=in_axis, out_axis=out_axis)
        assert jnp.max(jnp.abs(outputs_fast - outputs)) == 0

def test_mp_matmul():
    mesh = jax.sharding.Mesh(np.asarray(jax.devices("tpu")).reshape(1, -1), ("dp", "mp"))

    a, b, c, bs = 512, 256, 512, 32
    for order in (-1, 1):
        in_axis, out_axis = ("mp", None)[::order]
        quants = jax.random.randint(jax.random.PRNGKey(0), (a // bs, bs, b), 0, 255, dtype=jnp.int8)
        scale = jax.random.normal(jax.random.PRNGKey(1), (a // bs, 1, b), dtype=jnp.bfloat16) / 255
        inputs = jax.random.normal(jax.random.PRNGKey(2), (c, a), dtype=jnp.bfloat16)
        
        quants_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(in_axis, None, out_axis))
        scale_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(in_axis, None, out_axis))
        inputs_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp", in_axis))
        quants = jax.device_put(quants, quants_sharding)
        scale = jax.device_put(scale, scale_sharding)
        inputs = jax.device_put(inputs, inputs_sharding)

        outputs = matmul_8bit(quants, scale, inputs)
        outputs_fast = matmul_8bit_fast(quants, scale, inputs, mesh, in_axis=in_axis, out_axis=out_axis)
        assert jnp.max(jnp.abs(outputs_fast - outputs)) == 0

def test_mp_dp_matmul():
    mesh = jax.sharding.Mesh(np.asarray(jax.devices("tpu")).reshape(-1, 2), ("dp", "mp"))

    a, b, c, bs = 512, 256, 512, 32
    for order in (-1, 1):
        in_axis, out_axis = ("mp", None)[::order]
        quants = jax.random.randint(jax.random.PRNGKey(0), (a // bs, bs, b), 0, 255, dtype=jnp.int8)
        scale = jax.random.normal(jax.random.PRNGKey(1), (a // bs, 1, b), dtype=jnp.bfloat16) / 255
        inputs = jax.random.normal(jax.random.PRNGKey(2), (c, a), dtype=jnp.bfloat16)
        
        quants_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(in_axis, None, out_axis))
        scale_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(in_axis, None, out_axis))
        inputs_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp", in_axis))
        quants = jax.device_put(quants, quants_sharding)
        scale = jax.device_put(scale, scale_sharding)
        inputs = jax.device_put(inputs, inputs_sharding)

        outputs = matmul_8bit(quants, scale, inputs)
        outputs_fast = matmul_8bit_fast(quants, scale, inputs, mesh, in_axis=in_axis, out_axis=out_axis)
        assert jnp.max(jnp.abs(outputs_fast - outputs)) == 0