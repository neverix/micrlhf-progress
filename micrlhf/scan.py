import jax
from penzai import pz


@pz.pytree_dataclass
class ScanSequential(pz.Layer):
    layer: pz.Layer

    # @jax.jit
    def __call__(self, inputs):
        return jax.lax.scan(
            lambda h, l: (l(h), None),
            inputs,
            jax.tree_map(lambda x: x.untag("layer"), self.layer))[0]


def sequential_to_scan(model, sequential_n=1, return_aux=True):
    aux = {}
    def fn(seq):
        nonlocal aux
        layers = seq.sublayers
        layers = [l for l in layers if not isinstance(l, pz.nn.Identity)]
        layer = jax.tree_map(lambda *xs: pz.nx.stack(xs, "layer"), *layers)
        folded = ScanSequential(layer)
        aux["n_layers"] = len(layers)
        return folded

    result = model.select().at_instances_of(pz.nn.Sequential).pick_nth_selected(sequential_n).apply(fn)
    if return_aux:
        return result, aux
    return result
