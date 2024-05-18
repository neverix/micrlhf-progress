from micrlhf.llama import LlamaBlock
from typing import Literal
from penzai import pz
import dataclasses
import jax


@pz.pytree_dataclass
class ActivationAblation(pz.Layer):
    addition: pz.nx.NamedArray
    position: Literal["all", "last", "first"] = dataclasses.field(metadata={"pytree_node": False}, default="all")
    size_cond: Literal["all", "lsat"] = dataclasses.field(metadata={"pytree_node": False}, default="all")
    
    def adder(self, a, b):
        b = b / (jnp.linalg.norm(b) + 1e-10)
        updated = a - b * (a * b).sum(-1, keepdims=True)
        if self.position == "all":
            return updated
        elif self.position == "last":
            return a.at[-1].set(updated[-1])
        elif self.position == "first":
            return a.at[0].set(updated[0])
    
    def __call__(self, x):
        return pz.nx.nmap(lambda a, b: jax.lax.select(
            self.size_cond == "all" or len(a) > 1, self.adder(a, b).astype(a), a))(
            x.untag("seq", "embedding"), self.addition.untag("embedding")).tag("seq", "embedding")


@pz.pytree_dataclass
class ActivationReplacement(pz.Layer):
    replacement: pz.nx.NamedArray
    position: int = dataclasses.field(metadata={"pytree_node": False})
    
    def __call__(self, x):
        return pz.nx.nmap(lambda a, b: a.at[self.position].set(b.astype(a.dtype)))(
            x.untag("seq"), self.replacement).tag("seq")


@pz.pytree_dataclass
class ActivationAddition(pz.Layer):
    addition: pz.nx.NamedArray
    position: Literal["all", "last", "first"] = dataclasses.field(metadata={"pytree_node": False}, default="all")
    size_cond: Literal["all", "lsat"] = dataclasses.field(metadata={"pytree_node": False}, default="all")
    
    def adder(self, a, b):
        if self.position == "all":
            return a + b
        elif self.position == "last":
            return a.at[-1].add(b)
        elif self.position == "first":
            return a.at[0].add(b)
    
    def __call__(self, x):
        return pz.nx.nmap(lambda a, b: jax.lax.select(
            self.size_cond == "all" or len(a) > 1, self.adder(a, b).astype(a), a))(
            x.untag("seq"), self.addition).tag("seq")


def ablate_direction(llama, direction):
    act_abl = llama.select().at_instances_of(LlamaBlock).apply_with_selected_index(
        lambda layer, x: pz.nn.Sequential(
            [
                ActivationAblation(pz.nx.wrap(direction, "embedding"), position="all", size_cond="all"),
                x
            ]))
    return act_abl

def add_vector(llama, vector, layer, scale=1.0, position="all", size_cond="all"):
    if vector.ndim == 2:
        vector = pz.nx.wrap(vector * scale, "batch", "embedding")
    else:
        pz.nx.wrap(vector * scale, "embedding")
    act_add = llama.select().at_instances_of(LlamaBlock).pick_nth_selected(layer).apply(
        lambda x: pz.nn.Sequential(
            [
                ActivationAddition(vector, position=position, size_cond=size_cond),
                x
            ]))
    return act_add

def replace_activation(llama, vector, positions=None, prompt=None, tokenizer=None, replace_token="X", layer=0):
    if positions is None:
        positions = [i for i, a in enumerate(tokenizer.encode(prompt)) if tokenizer.decode([a]) == replace_token]
    if vector.ndim == 2:
        vector = pz.nx.wrap(vector, "batch", "embedding")
    else:
        vector = pz.nx.wrap(vector, "embedding")
    act_rep = llama.select().at_instances_of(LlamaBlock).pick_nth_selected(layer).apply(
        lambda x: pz.nn.Sequential([ActivationReplacement(
            vector, position=position) for position in positions] + [x]))
    return act_rep
