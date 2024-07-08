import jax.numpy as jnp
import jax

def matmul_6bit(inputs, scales, ql, qh, sc):
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

    matrix = scales * jnp.concatenate([
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

    return inputs @ matrix.reshape(inputs.shape[-1], -1)


def gen_6bit_mat():
    a, b, c = 1024, 256, 32
    bs = 256
    key = jax.random.key(0)
    key_scales, key_ql, key_qh, key_sc = jax.random.split(key, 4)
    scales = jax.random.normal(key_scales, (a // bs, 1, b), dtype=jnp.float32)
    ql = jax.random.randint(key_ql, (a // bs, 128, b), 0, 255, dtype=jnp.int16)
    qh = jax.random.randint(key_qh, (a // bs, 64, b), 0, 255, dtype=jnp.int16)
    sc = jax.random.normal(key_sc, (a // bs, 16, b), dtype=jnp.float32)
    inputs = jax.random.normal(jax.random.PRNGKey(2), (c, a), dtype=jnp.float32)
    
    print(matmul_6bit(inputs, scales, ql, qh, sc).shape)


if __name__ == "__main__":
    gen_6bit_mat()
