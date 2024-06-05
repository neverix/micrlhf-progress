from functools import partial
from collections import OrderedDict

import jax
from penzai import pz
import equinox as eqx
import numpy as np
from penzai.core import struct


def is_nx(x):
    return isinstance(x, pz.nx.NamedArrayBase)


@struct.pytree_dataclass
class ConcatenatedArray(struct.Struct, pz.nx.NamedArrayBase):
    pass


@pz.pytree_dataclass
class ScanSequential(pz.Layer):
    layer: pz.Layer

    # @jax.jit
    def __call__(self, inputs):
        layer = jax.tree_map(
            lambda x: x.untag("layer") if is_nx(x) and "layer" in x.named_axes else x,
            self.layer,
            is_leaf=is_nx)
        layer_nx, layer_base = eqx.partition(layer, is_nx, is_leaf=is_nx)
        def scanner(h, l):
            l_ = eqx.combine(l, layer_base, is_leaf=is_nx)
            h_ = l_(h)
            h_ = pz.nx.wrap(h_.unwrap(*h.named_axes.keys()), *h.named_axes.keys())
            return h_, None
        return jax.lax.scan(
            scanner,
            inputs,
            layer_nx)[0]


def pick_sequential(m, n):
    seq = m.at_instances_of(pz.nn.Sequential).pick_nth_selected(n[0])
    if len(n) == 1:
        return seq
    return pick_sequential(seq.at_children(), n[1:])


def sequential_to_scan(model, sequential_n=(0, 0), return_aux=False):
    aux = {}
    # @partial(jax.jit, donate_argnums=(0,))
    def fn(seq):
        layers = seq.sublayers
        layers = [l for l in layers if not isinstance(l, pz.nn.Identity)]
        weightses, treedefs = zip(*[eqx.partition(l, is_nx, is_leaf=is_nx) for l in layers])
        ws, tds = zip(*[jax.tree_util.tree_flatten(w, is_leaf=is_nx) for w in weightses])
        # ws, tds = zip(*[jax.tree_util.tree_flatten(l, is_leaf=is_nx) for l in layers])
        
        # ws_ = []
        # for ww in zip(*ws):
        #     ws_.append(pz.nx.stack(ww, "layer"))
        #     import time
        #     time.sleep(2)
        # w = ws_
        
        # w = jax.tree.map(jax.jit(lambda *ws: pz.nx.stack(ws, "layer"), donate_argnums=list(range(len(ws)))), *ws, is_leaf=is_nx)
        
        w = jax.tree.map(lambda *ws: pz.nx.NamedArray(
            data_array=np.array([w.data_array for w in ws]),
            named_axes=ws[0].named_axes
        ), *ws, is_leaf=is_nx)
        # w = jax.tree.map(jax.jit(lambda *ws: SimpleNamespace(
        #     untag=lambda t: np.array(ws)),
        #     donate_argnums=list(range(len(ws)))), *ws, is_leaf=is_nx)
        
        weight = jax.tree_util.tree_unflatten(tds[0], w)
        layer = eqx.combine(weight, treedefs[0], is_leaf=is_nx)
        # layer = weight
        folded = ScanSequential(layer)
        aux["n_layers"] = len(layers)
        return folded

    result = pick_sequential(model.select(), sequential_n).apply(fn)
    if return_aux:
        return result, aux
    return result
