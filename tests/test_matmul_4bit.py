import jax.numpy as jnp
import jax


compute = lambda f: (f(jnp.arange(256, dtype=jnp.uint8)), (f(jnp.arange(256, dtype=jnp.uint8).astype(jnp.int8))).astype(jnp.uint8))

# https://github.com/99991/pygguf/blob/829886d0726c89c6f6c0d8c39b0d507ec1604077/gguf.py#L209
def matmul_4bit(inputs, scale_factors, scale_offsets, qs1, qs2, based=False):
    num_blocks = scale_factors.shape[0]
    assert num_blocks == scale_offsets.shape[0] == qs1.shape[0] == qs2.shape[0] == 8
    
    sr = jax.lax.shift_right_logical

    # if based:
    #     def switch(x):
    #         x = x.astype(jnp.float16)
    #         x = x.view(jnp.uint16)
    #         x = x.astype(jnp.uint32)
    #         # https://gist.github.com/milhidaka/95863906fe828198f47991c813dbe233
    #         print(x.dtype, x.shape)
    #         sign = sr(x, 15)
    #         exponent = sr(x, 10) & 0x1f
    #         fraction = x & 0x3ff

    #         exponent_is_zero = exponent == 0
    #         is_zero = exponent_is_zero & (fraction == 0)
    #         exponent = jnp.full_like(fraction, 127 - 14)
    #         loop_has_stopped = exponent_is_zero & ~is_zero
    #         for _ in range(10):
    #             loop_has_stopped = loop_has_stopped | ((fraction & (1 << 10)) == 0)
    #             exponent = jnp.where(loop_has_stopped, exponent, exponent - 1)
    #             fraction = jnp.where(loop_has_stopped, fraction, fraction << 1)
    #         fraction = fraction & 0x3ff

    #         value = jnp.where(exponent_is_zero, jnp.where(is_zero, sign << 31, (sign << 31) | (exponent << 23) | (fraction << 13)),
    #                           (sign << 31) | ((exponent + (127-15)) << 23) | (fraction << 13))
    #         return value
    # else:
    def switch(x):
        x = x.astype(jnp.float16)
        x = x.astype(jnp.float32) * 10
        return x
    scale_factors = switch(scale_factors)
    scale_offsets = switch(scale_offsets)
    if based:
        scale_factors = scale_factors.transpose(1, 0, 2).reshape(1, 1, num_blocks, -1).astype(jnp.float32)
        scale_offsets = scale_offsets.transpose(1, 0, 2).reshape(1, 1, num_blocks, -1).astype(jnp.float32)
        qs1 = qs1.transpose(1, 0, 2).reshape(12, 1, num_blocks, -1)
        qs2 = qs2.transpose(1, 0, 2).reshape(4, 32, num_blocks, -1)
    else:
        scale_factors = scale_factors.transpose(0, 2, 1).reshape(-1, 1, 1).astype(jnp.float32)
        scale_offsets = scale_offsets.transpose(0, 2, 1).reshape(-1, 1, 1).astype(jnp.float32)
        qs1 = qs1.transpose(0, 2, 1).reshape(-1, 12, 1)
        qs2 = qs2.transpose(0, 2, 1).reshape(-1, 4, 32)

    qs1 = qs1.astype(jnp.int32)
    qs2 = qs2.astype(jnp.int32)
    i8tou8 = lambda x: jnp.where(x < 0, 256 + x, x)
    qs1 = i8tou8(qs1)
    qs2 = i8tou8(qs2)


    # max 63
    if based:
        chunk1 = qs1[0:4]
        chunk2 = qs1[4:8]
        chunk3 = qs1[8:]
        factor_scale = jnp.concatenate([chunk1 & 0b111111, (chunk3 & 15) | (sr(chunk1, 6) << 4)], axis=0)
        offset_scale = jnp.concatenate([chunk2 & 0b111111, (sr(chunk3, 4) % 16) | (sr(chunk2, 6) << 4)], axis=0)
    else:
        chunk1 = qs1[:, 0:4]
        chunk2 = qs1[:, 4:8]
        chunk3 = qs1[:, 8:]
        factor_scale = jnp.concatenate([chunk1 & 0b111111, (chunk3 & 15) | (sr(chunk1, 6) << 4)], axis=1)
        offset_scale = jnp.concatenate([chunk2 & 0b111111, (sr(chunk3, 4) % 16) | (sr(chunk2, 6) << 4)], axis=1)

    basify = lambda x: x  # x.astype(jnp.int8) if not based else x
    factors = scale_factors * basify(factor_scale)
    offsets = scale_offsets * basify(offset_scale)

    # max 15
    if based:
        qs2 = jnp.stack([qs2 & 0xf, sr(qs2, 4)], axis=1).reshape(8, 32, num_blocks, -1)
    else:
        qs2 = jnp.stack([qs2 & 0xf, sr(qs2, 4)], axis=2).reshape(-1, 8, 32)

    matrix = factors * basify(qs2) - offsets
    if not based:
        matrix = matrix.reshape(num_blocks, -1, 256).transpose(0, 2, 1)
        return inputs @ matrix.reshape(inputs.shape[-1], -1)
    else:
        matrix = matrix.reshape(256, num_blocks, -1)
        inputs = inputs.reshape(inputs.shape[0], num_blocks, 256)
        result = jax.lax.dot_general(inputs, matrix, (((2, 1), (0, 1)), ((), ())))
        return result

def gen_4bit_mat():
    
    a, b, c = 256 * 8, 256, 512
    bs = 256
    key = jax.random.key(0)
    key_scale_factors, key_scale_offsets, key_qs1, key_qs2 = jax.random.split(key, 4)
    scale_factors = jax.random.normal(key_scale_factors, (a // bs, 1, b), dtype=jnp.bfloat16)
    scale_offsets = jax.random.normal(key_scale_offsets, (a // bs, 1, b), dtype=jnp.bfloat16)
    qs1 = jax.random.randint(key_qs1, (a // bs, 12, b), 0, 255, dtype=jnp.uint8).view(jnp.int8)
    qs2 = jax.random.randint(key_qs2, (a // bs, 128, b), 0, 255, dtype=jnp.uint8).view(jnp.int8)
    inputs = jax.random.normal(jax.random.PRNGKey(2), (c, a), dtype=jnp.bfloat16)
    
    d = matmul_4bit(inputs, scale_factors, scale_offsets, qs1, qs2)
    print(d)
    e = matmul_4bit(inputs, scale_factors, scale_offsets, qs1, qs2, based=True)
    print(d - e)
    print(jnp.abs(d - e).max())


if __name__ == "__main__":
    gen_4bit_mat()
