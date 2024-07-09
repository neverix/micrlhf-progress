from optax import GradientTransformation, scale_by_adam, scale_by_learning_rate, chain
from matplotlib import pyplot as plt
import jax
import jax.numpy as jnp
from tqdm.auto import trange
import equinox as eqx
import numpy as np


def gen_quant_table(dtq=False, signed=True):
    numbers = []
    if dtq:
        values = []
        for sign in ((-1, 1) if signed else (1,)):
            max_digits = 7 if signed else 8
            for indicator in range(max_digits, -1, -1):
                digits = max(0, max_digits - 1 - indicator)
                for remainder in range(2 ** digits):
                    number = []
                    if signed:
                        number += [(sign + 1) // 2]
                    number += [0] * indicator
                    number += [1]
                    number += list(map(int, bin(remainder)[2:].zfill(digits)))
                    value = sign * (10 ** -indicator) * ((remainder if digits else 1) / 2 ** digits)
                    values.append((number, value))
        for _, value in sorted(values):
            numbers.append(value)
    else:
        if signed:
            for sign in (-1, 1):
                for i in range(128):
                    numbers.append(sign * i / 127)
        else:
            for i in range(256):
                numbers.append(i / 255)
    return numbers

def scale_by_adam_8bit(
    b1: float = 0.9,
    b2: float = 0.99,
    eps: float = 1e-8,
    block_size: int = 32,
    dtq: bool = True,
    dtype: jax.typing.DTypeLike = jnp.float32,
):
    # TODO
    # special treatment for adagrad
    # atan2 adam
    # kernel for update

    def sort_find_indices(x):
        x = jnp.asarray(x, dtype=dtype)
        return jnp.sort(x), jnp.argsort(x)
    quant_table_signed = gen_quant_table(dtq=dtq, signed=True)
    quant_table_unsigned = gen_quant_table(dtq=dtq, signed=False)
    dequant_array_signed = jnp.asarray(quant_table_signed, dtype=dtype)
    dequant_array_unsigned = jnp.asarray(quant_table_unsigned, dtype=dtype)
    quant_search_signed, quant_index_signed = sort_find_indices(quant_table_signed)
    quant_search_unsigned, quant_index_unsigned = sort_find_indices(quant_table_unsigned)
    adagrad_mode = b1 == 0.0

    def quant_array(value, signed=True):
        quanted = jnp.searchsorted(quant_search_signed if signed else quant_search_unsigned, value)
        quanted_indexed = (quant_index_signed if signed else quant_index_unsigned)[quanted]
        return quanted_indexed
    def dequant_array(value, signed=True):
        array = dequant_array_signed if signed else dequant_array_unsigned
        return array[value]

    def flatten(value):
        leaves, treedef = jax.tree.flatten(value, is_leaf=eqx.is_array)
        shapes, flats = [x.shape for x in leaves], [x.reshape(-1, block_size if x.size >= block_size and len(x.shape) > 1 else 1) for x in leaves]
        return (treedef, shapes), flats
    def quantize(flats, signed=True):
        scales = [jnp.abs(x).max(-1, keepdims=True) for x in flats]
        quants = [quant_array(x / s, signed=signed) for x, s in zip(flats, scales)]
        return quants, scales
    def unflatten(shapedef, value):
        unflats = [x.reshape(s) for x, s in zip(value, shapedef[1])]
        return jax.tree.unflatten(shapedef[0], unflats)
    def dequantize(quantized, signed=True):
        return [dequant_array(quants, signed=signed) * scales for quants, scales in zip(*quantized)]

    def init(x):
        shapedef, _ = flatten(x)
        _, shapes = shapedef
        momentum_quants, momentum_scales, norm_quants, norm_scales = [], [], [], []
        init_mom = quant_array(0.0, signed=True)
        init_scale = quant_array(0.0, signed=False)
        for shape in shapes:
            flattened = np.prod(shape)
            bs = block_size if flattened >= block_size and len(shape) > 1 else 1
            shape_quant = (flattened // bs, bs)
            shape_scale = (flattened // bs, 1)
            if not adagrad_mode:
                momentum_quants.append(np.full(shape_quant, init_mom, dtype=np.uint8))
                momentum_scales.append(np.ones(shape_scale, dtype=dtype))
            norm_quants.append(np.full(shape_quant, init_scale, dtype=np.uint8))
            norm_scales.append(np.ones(shape_scale, dtype=dtype))
        return (None if adagrad_mode else (momentum_quants, momentum_scales)), (norm_quants, norm_scales), jnp.array(0, dtype=jnp.uint32)
    def update(grad, state, params=None):
        shapedef, grad = flatten(grad)
        momentum, norm, count = state
        norm = dequantize(norm, signed=False)
        if not adagrad_mode:
            momentum = dequantize(momentum, signed=True)
            momentum = jax.tree.map(lambda g, m: b1 * m + (1 - b1) * g, grad, momentum)
        norm = jax.tree.map(lambda g, n: b2 * n + (1 - b2) * g ** 2, grad, norm)
        count = count + 1
        update = jax.tree.map(lambda m, n: (m / ((1 - b1 ** count) if not adagrad_mode else 1)) / (jnp.sqrt(n / (1 - b2 ** count)) + eps), momentum if not adagrad_mode else grad, norm)
        return unflatten(shapedef, update), (None if adagrad_mode else quantize(momentum), quantize(norm, signed=False), count)
    return GradientTransformation(init, update)


def transpose():
    def init(x):
        return
    def update(grad, state, params=None):
        return jax.tree.map(lambda g: jnp.swapaxes(g, -1, -2), grad), state
    return GradientTransformation(init, update)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import os

    os.makedirs("figures/scratch", exist_ok=True)
    
    standard_signed = gen_quant_table(dtq=False, signed=True)
    standard_unsigned = gen_quant_table(dtq=False, signed=False)
    dtq_signed = gen_quant_table(dtq=True, signed=True)
    dtq_unsigned = gen_quant_table(dtq=True, signed=False)
    
    # logarithmic signed bins
    b = [c * 10 ** i for i in range(-8, 0) for c in (1, 2, 5, 9)]
    b = [-a for a in b] + b
    b = sorted(b)
    
    plt.hist(standard_signed, bins=b, alpha=0.25, label="Standard Signed")
    plt.hist(standard_unsigned, bins=b, alpha=0.25, label="Standard Unsigned")
    plt.hist(dtq_signed, bins=b, alpha=0.25, label="DTQ Signed")
    plt.hist(dtq_unsigned, bins=b, alpha=0.25, label="DTQ Unsigned")
    plt.legend()
    plt.savefig("figures/scratch/quant_table.png")
    plt.clf()
    
    k = 512
    n_iter = 512
    for optimizer, fn in ((chain(scale_by_adam_8bit(b1=0.0, b2=0.99), scale_by_learning_rate(5e-3)), "8bit"), (chain(scale_by_adam(b1=0.0, b2=0.99), scale_by_learning_rate(5e-3)), "basic")):
        key = jax.random.PRNGKey(0)
        w_key, x_key, w_key_ = jax.random.split(key, 3)
        w = jax.random.normal(w_key, (k, k)) / (k ** 0.5)
        x = jax.random.normal(x_key, (k, k))
        y = x @ w
        b = jax.random.normal(w_key, (1,))
        w_ = jax.random.normal(w_key_, (k, k))
        p = [w_, b]
        opt_state = optimizer.init(p)
        
        def update(opt_state, p):
            loss, grad = jax.value_and_grad(lambda p, x, y: jnp.mean((x @ p[0] + p[1] - y) ** 2))(p, x, y)
            p, opt_state = optimizer.update(grad, opt_state, p)
            return loss, p, opt_state
        update = jax.jit(update, donate_argnums=(0, 1))
        
        losses = []
        for _ in (bar := trange(n_iter)):
            loss, p, opt_state = update(opt_state, p)
            losses.append(loss)
            bar.set_postfix(loss=loss)

        plt.plot(losses, label=fn)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-1, 1)
    plt.legend()
    plt.savefig("figures/scratch/optim.png")
