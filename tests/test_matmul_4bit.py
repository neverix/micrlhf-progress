import jax.numpy as jnp
import jax


compute = lambda f: (f(jnp.arange(256, dtype=jnp.uint8)), (f(jnp.arange(256, dtype=jnp.uint8).astype(jnp.int8))).astype(jnp.uint8))

# https://github.com/99991/pygguf/blob/829886d0726c89c6f6c0d8c39b0d507ec1604077/gguf.py#L209
def matmul_4bit(inputs, scale_factors, scale_offsets, qs1, qs2, use_int8=False):
    num_blocks = scale_factors.shape[0]
    assert num_blocks == scale_offsets.shape[0] == qs1.shape[0] == qs2.shape[0]

    scale_factors = scale_factors.transpose(0, 2, 1).reshape(-1, 1, 1)
    scale_offsets = scale_offsets.transpose(0, 2, 1).reshape(-1, 1, 1)
    qs1 = qs1.transpose(0, 2, 1).reshape(-1, 12, 1)
    qs2 = qs2.transpose(0, 2, 1).reshape(-1, 4, 32)

    qs1 = qs1.astype(jnp.int32)
    qs2 = qs2.astype(jnp.int32)
    i8tou8 = lambda x: jnp.where(x < 0, 256 + x, x)
    qs1 = i8tou8(qs1)
    qs2 = i8tou8(qs2)

    chunk1 = qs1[:, 0:4]
    chunk2 = qs1[:, 4:8]
    chunk3 = qs1[:, 8:]

    factor_scale = jnp.concatenate([chunk1 & 0b111111, (chunk3 & 15) | ((chunk1 >> 6) << 4)], axis=1)
    offset_scale = jnp.concatenate([chunk2 & 0b111111, ((chunk3 >> 4) % 16) | ((chunk2 >> 6) << 4)], axis=1)

    factors = scale_factors * factor_scale.astype(jnp.int8)
    offsets = scale_offsets * offset_scale.astype(jnp.int8)

    qs2 = jnp.stack([qs2 & 0xf, qs2 >> 4], axis=2).reshape(-1, 8, 32)

    matrix = factors * qs2.astype(jnp.int8) - offsets
    matrix = matrix.reshape(num_blocks, -1, 256).transpose(0, 2, 1)
    return inputs @ matrix.reshape(inputs.shape[-1], -1)

def gen_4bit_mat():
    
    a, b, c = 1024, 256, 32
    bs = 256
    key = jax.random.key(0)
    key_scale_factors, key_scale_offsets, key_qs1, key_qs2 = jax.random.split(key, 4)
    scale_factors = jax.random.normal(key_scale_factors, (a // bs, 1, b), dtype=jnp.bfloat16)
    scale_offsets = jax.random.normal(key_scale_offsets, (a // bs, 1, b), dtype=jnp.bfloat16)
    qs1 = jax.random.randint(key_qs1, (a // bs, 12, b), 0, 255, dtype=jnp.uint8).view(jnp.int8)
    qs2 = jax.random.randint(key_qs2, (a // bs, 128, b), 0, 255, dtype=jnp.uint8).view(jnp.int8)
    inputs = jax.random.normal(jax.random.PRNGKey(2), (c, a), dtype=jnp.bfloat16)
    
    d = matmul_4bit(inputs, scale_factors, scale_offsets, qs1, qs2)
    e = matmul_4bit(inputs, scale_factors, scale_offsets, qs1, qs2, use_int8=True)
    print(d - e)
    print(jnp.abs(d - e).max())


if __name__ == "__main__":
    gen_4bit_mat()
