import dataclasses

import jax
import jax.experimental.pallas.ops.tpu.flash_attention
import jax.experimental.shard_map
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from penzai import pz

from .llama import LlamaAttention


def common_axes(*named_arrays):
    shape = named_arrays[0].named_shape
    for na in named_arrays[1:]:
        for k, v in list(shape.items()):
            if v != na.named_shape.get(k):
                del shape[k]
    return shape.keys()


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class LlamaFlashAttention(pz.nn.Attention):
    out_proj: pz.LayerLike

    seq_axis: str = dataclasses.field(metadata={"pytree_node": False})
    kv_seq_axis: str = dataclasses.field(metadata={"pytree_node": False})
    head_axis: str = dataclasses.field(metadata={"pytree_node": False})
    projection_axis: str = dataclasses.field(metadata={"pytree_node": False})

    mask_value: pz.de.SideInputRequest
    mask_with: jax.typing.ArrayLike

    mesh_request: pz.de.SideInputRequest
    axis_name_to_mesh_name_request: pz.de.SideInputRequest

    def __call__(self, x: pz.nx.NamedArray) -> pz.nx.NamedArray:
        query = self.input_to_query(x)
        key = self.input_to_key(x)
        value = self.input_to_value(x)
        batch_axes = list(set(common_axes(query, key, value))
                          - {self.seq_axis, self.head_axis, self.projection_axis})
        q, k, v = (pz.nx.nmap(lambda x: x.flatten())(x.untag(*batch_axes)).tag("batch") for x in (query, key, value))
        q, k, v = (x.untag("batch", self.head_axis, self.seq_axis, self.projection_axis) for x in (q, k, v))
        attn_bias = pz.nx.nmap(lambda mv: (~mv) * self.mask_with)(self.mask_value.ask())
        batch_size, num_heads, *_ = q.positional_shape
        ab = pz.nx.nmap(lambda x:
            jnp.repeat(jnp.repeat(x[None, None], batch_size, 0), num_heads, 1)
            )(attn_bias).tag("batch", self.head_axis)
        ab = ab.untag("batch", self.head_axis, self.seq_axis, self.kv_seq_axis)

        mesh, axis_name_to_mesh_name = self.mesh_request.ask(), dict(self.axis_name_to_mesh_name_request.ask())
        o = pz.nx.nmap(lambda q, k, v, ab:
            jax.experimental.shard_map.shard_map((
                lambda q, k, v, ab: jax.experimental.pallas.ops.tpu.flash_attention.flash_attention(
                    q, k, v, ab=ab
                )
            ),
            mesh=mesh,
            # TODO what to do with batch?
            in_specs=(P(axis_name_to_mesh_name.get("batch"),
                axis_name_to_mesh_name.get(self.head_axis),
                axis_name_to_mesh_name.get(self.seq_axis),
                # splitting apart the head dimension is just silly
                None),) * 3
            + (P(axis_name_to_mesh_name.get("batch"),
                 axis_name_to_mesh_name.get(self.head_axis),
                 # no KV seq split 😔
                 None,
                 axis_name_to_mesh_name.get(self.seq_axis)
                 ),),
            out_specs=P(axis_name_to_mesh_name.get("batch"),
               axis_name_to_mesh_name.get(self.head_axis),
               axis_name_to_mesh_name.get(self.seq_axis),
               None),
            check_rep=False)(q, k, v, ab),
        )(q, k, v, ab)

        output = o.tag("batch", self.head_axis, self.seq_axis, self.projection_axis)
        output = self.out_proj(output)

        return output

    @classmethod
    def from_regular(
        cls,
        original: LlamaAttention,
        seq_axis: str = "seq",
        kv_seq_axis: str = "kv_seq",
        head_axis: str = "kv_heads",
        projection_axis: str = "projection",
        mesh_tag: str = "mesh",
        axis_name_to_mesh_name_tag="axis_name_to_mesh_name"
    ) -> "LlamaFlashAttention":
        masker = original.query_key_to_attn.select() \
            .at_instances_of(pz.nn.ApplyAttentionMask).pick_nth_selected(0).get()
        out_proj = original.attn_value_to_output.select() \
            .at_instances_of(pz.nn.Linear).pick_nth_selected(0).get()
        
        return cls(
            input_to_query=original.input_to_query,
            input_to_key=original.input_to_key,
            input_to_value=original.input_to_value,
            out_proj=out_proj,
            
            mask_value=masker.mask,
            mask_with=masker.masked_out_value,
            
            seq_axis=seq_axis,
            kv_seq_axis=kv_seq_axis,
            head_axis=head_axis,
            projection_axis=projection_axis,
            
            mesh_request=pz.de.SideInputRequest(tag=mesh_tag),
            axis_name_to_mesh_name_request=pz.de.SideInputRequest(tag=axis_name_to_mesh_name_tag),
            
            query_key_to_attn=None,
            attn_value_to_output=None,
        )


def flashify(model):
    model = model.select().at_instances_of(LlamaAttention).apply(
        lambda a: LlamaFlashAttention.from_regular(a)
    )
    model = model.handle_sharding(mod=LlamaFlashAttention)
    return model
