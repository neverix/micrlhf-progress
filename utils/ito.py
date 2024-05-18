import jax
import jax.numpy as jnp


def matched_pursuit_update_step(residual, weights, dictionary):
    inner_products = jnp.einsum("fv,v->f", dictionary, residual)
    idx = jnp.argmax(jnp.abs(inner_products))
    # inner_products = jnp.abs(inner_products); idx = jnp.argmax(inner_products)
    a = inner_products[idx]
    residual = residual - a * dictionary[idx]
    weights = weights.at[idx].set(a)
    return residual, weights

def matched_pursuit(signal, dictionary, target_l0):
    residual = signal
    weights = jnp.zeros(dictionary.shape[0])
    def mpus(rw, _):
        return matched_pursuit_update_step(*rw, dictionary), None
    (residual, weights), _ = jax.lax.scan(mpus, (residual, weights), None, target_l0)
    reconstruction = jnp.einsum("fv,f->v", dictionary, weights)
    return weights, reconstruction

def grad_pursuit_update_step(signal, weights, dictionary, pos_only=True):
    residual = signal - jnp.einsum('fv,f->v', dictionary, weights)
    selected_features = (weights != 0)
    inner_products = jnp.einsum('fv,v->f', dictionary, residual)
    idx = jnp.argmax(inner_products) if pos_only else jnp.argmax(jnp.abs(inner_products))
    selected_features = selected_features.at[idx].set(True)
    
    grad = selected_features * inner_products
    c = jnp.einsum('f,fv->v', grad, dictionary)
    step_size = jnp.einsum('v,v->', c, residual) / jnp.einsum('v,v->', c, c)
    weights = weights + step_size * grad
    weights = jnp.maximum(weights, 0) if pos_only else weights
    return weights

def grad_pursuit(signal, dictionary, target_l0, initial=None, pos_only=True):
    if initial is None:
        weights = jnp.zeros(dictionary.shape[0])
    else:
        weights = initial
    # no thank you, I am a proud TPU user
    def gpus(w, _):
        return grad_pursuit_update_step(signal, w, dictionary, pos_only=pos_only), None
    weights, _ = jax.lax.scan(gpus, weights, None, target_l0)
    reconstruction = jnp.einsum("fv,f->v", dictionary, weights)
    return weights, reconstruction
