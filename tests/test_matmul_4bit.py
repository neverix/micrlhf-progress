import jax.numpy as jnp
import jax


# https://github.com/99991/pygguf/blob/829886d0726c89c6f6c0d8c39b0d507ec1604077/gguf.py#L209
def matmul_4bit(inputs, scale_factors, scale_offsets, qs1, qs2):
    num_blocks = scale_factors.shape[0]
    assert num_blocks == scale_offsets.shape[0] == qs1.shape[0] == qs2.shape[0]

    scale_factors = scale_factors.reshape(num_blocks, 1, 1, -1)
    scale_offsets = scale_offsets.reshape(num_blocks, 1, 1, -1)
    qs1 = qs1.reshape(num_blocks, 12, 1, -1)
    qs2 = qs2.reshape(num_blocks, 4, 32, -1)

    factors = scale_factors * jnp.concatenate([qs1[:, 0:4] & 0b111111, (qs1[:, 8:] & 15) | ((qs1[:, 0:4] >> 6) << 4)], axis=1)
    offsets = scale_offsets * jnp.concatenate([qs1[:, 4:8] & 0b111111, (qs1[:, 8:] >> 4) | ((qs1[:, 4:8] >> 6) << 4)], axis=1)
    
    qs2 = jnp.stack([qs2 & 0xf, qs2 >> 4], axis=2).reshape(num_blocks, 8, 32, -1)

    matrix = factors * qs2 - offsets
    print(matrix.shape)
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
    
    print(matmul_4bit(inputs, scale_factors, scale_offsets, qs1, qs2).shape)


if __name__ == "__main__":
    gen_4bit_mat()
