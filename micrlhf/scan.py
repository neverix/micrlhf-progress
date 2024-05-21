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


def sequential_to_scan(model, sequential_n=1):
    def fn(seq):
        layers = seq.sublayers
        layers = [l for l in layers if not isinstance(l, pz.nn.Identity)]
        layer = jax.tree_map(lambda *xs: pz.nx.stack(xs, "layer"), *layers)
        return ScanSequential(layer)

    return model.select().at_instances_of(pz.nn.Sequential).pick_nth_selected(sequential_n).apply(fn)
