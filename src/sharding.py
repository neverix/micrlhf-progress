import dataclasses
import typing
from typing import Any, Dict, Generic

import jax
from jax import sharding as jshard
from penzai import pz
from penzai.core import layer as layer_base
from penzai.core import selectors
from penzai.data_effects import effect_base
from penzai.data_effects.side_input import Tag
from penzai.toolshed import sharding_util

_T = typing.TypeVar("_T")

@pz.pytree_dataclass
class ConstrainedSharding(pz.Layer):
    mesh: pz.de.SideInputRequest[jshard.Mesh]
    axis_name_to_mesh_name: pz.de.SideInputRequest[Dict[str, str]]

    @classmethod
    def from_config(cls, mesh_tag="mesh", axis_name_to_mesh_name_tag="axis_name_to_mesh_name"):
        return cls(
            mesh=pz.de.SideInputRequest(mesh_tag),
            axis_name_to_mesh_name=pz.de.SideInputRequest(axis_name_to_mesh_name_tag),
        )

    def __call__(self, tree):
        mesh, axis_name_to_mesh_name = self.mesh.ask(), dict(self.axis_name_to_mesh_name.ask())
        return jax.lax.with_sharding_constraint(
            tree,
            sharding_util.name_to_name_sharding(tree, mesh, axis_name_to_mesh_name),
        )


@dataclasses.dataclass(frozen=True)
class SideInputEffectImplNonPytree(Generic[_T], effect_base.EffectRuntimeImpl):
  """Implementation of the side input effect."""

  _value: _T = dataclasses.field(metadata={"pytree_node": False})
  _handler_id: effect_base.HandlerId

  def ask(self) -> _T:
    return self._value

  def handler_id(self) -> effect_base.HandlerId:
    return self._handler_id

  @classmethod
  def effect_protocol(cls):
    return pz.de.SideInputEffect


@pz.pytree_dataclass
class WithConstantSideInputsNonPytree(effect_base.EffectHandler):
  """`SideInput` handler that provides side inputs using its own attribute.

  Attributes:
    handler_id: The ID of this handler.
    body: The layer that this handler wraps.
    side_inputs: The value for the side inputs that the handler provides.
  """

  handler_id: effect_base.HandlerId = dataclasses.field(
      metadata={"pytree_node": False}
  )
  body: layer_base.LayerLike
  side_inputs: dict[Tag, Any] = dataclasses.field(metadata={"pytree_node": False})

  @classmethod
  def effect_protocol(cls):
    return pz.de.SideInputEffect

  def __call__(self, argument: tuple[Any, Any]):
    impls = {
        tag: SideInputEffectImplNonPytree(_value=val, _handler_id=self.handler_id)
        for tag, val in self.side_inputs.items()
    }
    handled_body = (
        selectors.select(self.body)
        .at_instances_of(pz.de.HandledSideInputRef)
        .where(lambda ref: ref.handler_id == self.handler_id)
        .apply(lambda ref: impls[ref.tag])
    )
    return handled_body(argument)

  @classmethod
  def handling(
      cls,
      body: layer_base.LayerLike,
      side_inputs: dict[Tag, Any],
      handler_id: str | None = None,
      keep_unused: bool = False,
  ) -> "WithConstantSideInputsNonPytree":
    """Builds a ``WithConstantSideInputsNonPytree`` that handles effects in this layer.

    Args:
      body: The layer to wrap. Usually will contain `SideInputRequest` nodes.
      side_inputs: The constant values to provide for each tag that we should
        handle.
      handler_id: ID to use for the handler. If None, will be inferred.
      keep_unused: Whether to keep unused side inputs. If False, then any tag
        that isn't actually used by a `SideInputRequest` in the layer will be
        omitted from the handler's attributes.

    Returns:
      A ``WithConstantSideInputsNonPytree`` handler wrapping ``body``, with its side
      input holes with the given tag replaced with references to this handler.
    """
    handler_id = effect_base.infer_or_check_handler_id(
        "WithConstantSideInputsNonPytree", body, explicit_id=handler_id
    )
    selected_requests = (
        selectors.select(body)
        .at_instances_of(pz.de.SideInputRequest)
        .where(lambda req: req.tag in side_inputs)
    )
    used_tags = set()
    for req in selected_requests.get_sequence():
      used_tags.add(req.tag)

    if keep_unused:
      side_inputs_attr = dict(side_inputs)
    else:
      side_inputs_attr = {
          tag: val for tag, val in side_inputs.items() if tag in used_tags
      }
    return cls(
        handler_id=handler_id,
        body=selected_requests.apply(
            lambda req: pz.de.HandledSideInputRef(handler_id=handler_id, tag=req.tag)
        ),
        side_inputs=side_inputs_attr,
    )
