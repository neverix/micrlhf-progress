from optax import GradientTransformation, scale_by_adam, scale_by_learning_rate, chain
from matplotlib import pyplot as plt
import jax
import jax.numpy as jnp
from tqdm.auto import trange


def scale_by_adam_8bit(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-12,
    block_size: int = 256,
):
    def init(x):
        # momentum (int8)
        # momentum max (float32, block)
        # norm (int8)
        # norm max (float32, block)
        pass
    def update(grad, state, x):
        # update momentum
        # update norm
        # momentum bias corection
        # norm bias correction
        # tree_map m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat
        pass
    return GradientTransformation(init, update)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import os


    os.makedirs("figures/scratch", exist_ok=True)
    
    k = 2048
    key = jax.random.PRNGKey(0)
    w_key, x_key, w_key_ = jax.random.split(key, 3)
    w = jax.random.normal(w_key, (k, k)) / (k ** 0.5)
    x = jax.random.normal(x_key, (k, k))
    y = x @ w
    w_ = jax.random.normal(w_key_, (k, k))
    optimizer = chain(scale_by_adam(b1=0.0), scale_by_learning_rate(5e-3))
    opt_state = optimizer.init(w_)
    
    def update(opt_state, w_):
        loss, grad = jax.value_and_grad(lambda w, x, y: jnp.mean((x @ w - y) ** 2))(w_, x, y)
        w_, opt_state = optimizer.update(grad, opt_state, w_)
        return loss, w_, opt_state
    update = jax.jit(update, donate_argnums=(0, 1))
    
    losses = []
    for _ in (bar := trange(4096)):
        loss, w_, opt_state = update(opt_state, w_)
        losses.append(loss)
        bar.set_postfix(loss=loss)

    plt.plot(losses)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-1, 1)
    plt.savefig("figures/scratch/optim.png")
