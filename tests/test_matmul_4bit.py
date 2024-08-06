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
    if use_int8:
        qs1 = qs1.astype(jnp.int8).astype(jnp.int32)
        qs2 = qs2.astype(jnp.int8).astype(jnp.int32)

    factor_scale = jnp.concatenate([qs1[:, 0:4] & 0b111111, (qs1[:, 8:] & 15) | (((qs1[:, 0:4] >> 6) % 4) << 4)], axis=1)
    offset_scale = jnp.concatenate([qs1[:, 4:8] & 0b111111, ((qs1[:, 8:] >> 4) % 16) | (((qs1[:, 4:8] >> 6) % 4) << 4)], axis=1)

    factors = scale_factors * factor_scale
    offsets = scale_offsets * offset_scale

    qs2 = jnp.stack([qs2 & 0xf, (qs2 >> 4) % 16], axis=2).reshape(-1, 8, 32)

    matrix = factors * qs2 - offsets
    matrix = matrix.reshape(num_blocks, -1, 256).transpose(0, 2, 1)
    print({k: (v.shape if hasattr(v, "shape") else None) for k, v in locals().items()})
    return inputs @ matrix.reshape(inputs.shape[-1], -1)

def gen_4bit_mat():
    
    a, b, c = 1024, 256, 32
    bs = 256
    key = jax.random.key(0)
    key_scale_factors, key_scale_offsets, key_qs1, key_qs2 = jax.random.split(key, 4)
    scale_factors = jax.random.normal(key_scale_factors, (a // bs, 1, b), dtype=jnp.float16)
    scale_offsets = jax.random.normal(key_scale_offsets, (a // bs, 1, b), dtype=jnp.float16)
    qs1 = jax.random.randint(key_qs1, (a // bs, 12, b), 0, 255, dtype=jnp.uint8)
    qs2 = jax.random.randint(key_qs2, (a // bs, 128, b), 0, 255, dtype=jnp.uint8)
    inputs = jax.random.normal(jax.random.PRNGKey(2), (c, a), dtype=jnp.float16)
    
    d = matmul_4bit(inputs, scale_factors, scale_offsets, qs1, qs2)
    e = matmul_4bit(inputs, scale_factors, scale_offsets, qs1, qs2, use_int8=True)
    print(d - e)


if __name__ == "__main__":
    gen_4bit_mat()
