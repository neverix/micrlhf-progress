@dataclasses.dataclass
class LocalStateEffectImpl(Generic[_T], effect_base.EffectRuntimeImpl):
  """Implementation of the local state effect.

  Attributes:
    _state: Mutable (!) state tracked by the implementation.
    _handler_id: ID of the handler that is managing this state.
  """

  _state: _T
  _handler_id: effect_base.HandlerId

  def handler_id(self) -> effect_base.HandlerId:
    return self._handler_id

  @classmethod
  def effect_protocol(cls):
    return LocalStateEffect

  def get(self) -> _T:
    return self._state

  def set(self, value: _T):
    self._state = value


@struct.pytree_dataclass
class WithFunctionalLocalState(effect_base.EffectHandler):
  """`LocalState` effect handler that functionalizes local states.

  ``WithFunctionalLocalState`` transforms the body layer so that it takes a
  dictionary of states as an argument and returns a dictionary of states as a
  result.

  The standard way to construct a ``WithFunctionalLocalState`` handler is to use
  `handle_local_states`, which returns a functional wrapper and also the
  initial state callable. Conversely, you can re-embed local states into the
  model using `freeze_local_states`.
  """

  handler_id: effect_base.HandlerId = dataclasses.field(
      metadata={"pytree_node": False}
  )
  body: layer_base.LayerLike

  @layer_base.checked_layer_call
  def __call__(
      self, argument: tuple[Any, dict[str, Any]]
  ) -> tuple[Any, dict[str, Any]]:
    inner_arg, states = argument
    impls = {
        k: LocalStateEffectImpl(_state=v, _handler_id=self.handler_id)
        for k, v in states.items()
    }
    handled_body = (
        selectors.select(self.body)
        .at_instances_of(HandledLocalStateRef)
        .where(lambda ref: ref.handler_id == self.handler_id)
        .apply(lambda ref: impls[ref.name])
    )
    result = handled_body(inner_arg)
    new_states = {k: impl._state for k, impl in impls.items()}
    return result, new_states

  def _state_structure(self, desc):
    result = {}
    refs = (
        selectors.select(self.body)
        .at_instances_of(HandledLocalStateRef)
        .where(lambda ref: ref.handler_id == self.handler_id)
        .get_sequence()
    )
    for i, ref in enumerate(refs):
      result[ref.name] = shapecheck.Wildcard(f"{desc} {i}")
    return result

  def input_structure(self):
    return (
        shapecheck.Wildcard("input to body"),
        self._state_structure("old state"),
    )

  def output_structure(self):
    return (
        shapecheck.Wildcard("output from body"),
        self._state_structure("new state"),
    )

  @classmethod
  def effect_protocol(cls):
    return LocalStateEffect


@typing.overload
def handle_local_states(
    body: layer_base.LayerLike,
    category: Category | None = None,
    category_predicate: Callable[[Category], bool] | None = None,
    lazy: Literal[False] = False,
    state_sharing: Literal["forbidden", "allowed", "unsafe"] = "forbidden",
    handler_id: str | None = None,
) -> tuple[WithFunctionalLocalState, dict[str, Any]]:
  ...


@typing.overload
def handle_local_states(
    body: layer_base.LayerLike,
    category: Category | None = None,
    category_predicate: Callable[[Category], bool] | None = None,
    lazy: Literal[True] = False,  # pytype: disable=annotation-type-mismatch
    state_sharing: Literal["forbidden", "allowed", "unsafe"] = "forbidden",
    handler_id: str | None = None,
) -> tuple[WithFunctionalLocalState, Callable[[], dict[str, Any]]]:
  ...


def handle_local_states(
    body: layer_base.LayerLike,
    category: Category | None = None,
    category_predicate: Callable[[Category], bool] | None = None,
    lazy: bool = False,
    state_sharing: Literal["forbidden", "allowed", "unsafe"] = "forbidden",
    handler_id: str | None = None,
) -> tuple[
    WithFunctionalLocalState, dict[str, Any] | Callable[[], dict[str, Any]]
]:
  """Extracts local states from a stateful model.

  This method the primary way to transform a stateful model into a functional
  form that can be run.

  Args:
    body: A model or submodel with local state.
    category: The category of states to extract. Not needed if
      category_predicate is provided.
    category_predicate: An optional callable that returns True for categories we
      should take ownership of. Note that states with different categories must
      still have unique names if they are being handled by the same handler. Not
      needed if category is provided.
    lazy: If True, returns a callable that initializes the state, instead of
      returning the state itself.
    state_sharing: Strictness for sharing of states. If "forbidden", state
      sharing is strictly not allowed. If "allowed", state sharing is allowed
      between `InitialLocalStateRequest` states with identical initializers, and
      between `SharedLocalStateRequest` and any other state with the same name.
      If "unsafe", any states with the same name will be shared, with the value
      coming from whichever one was seen last.
    handler_id: ID to use for the handler. If None, will be inferred.

  Returns:
    A handler wrapping the model to handle the given states, and an initial
    state dictionary to pass as the second argument to that handler (or a
    callable producing that dictionary if `lazy` was True).
  """
  handler_id = effect_base.infer_or_check_handler_id(
      "WithFunctionalLocalState", body, explicit_id=handler_id
  )
  initial_state_thunks = {}

  if category is None and category_predicate is None:
    raise ValueError(
        "One of `category` and `category_predicate` must be specified. (If you"
        " want to handle states whose category is exactly None, use a category"
        " predicate `lambda x: x is None`.)"
    )
  elif category is not None and category_predicate is not None:
    raise ValueError(
        "Only one of `category` and `category_predicate` may be specified."
    )
  elif category_predicate is None:
    category_predicate = lambda c: c == category

  def _make_ref(
      keypath,
      hole: (
          InitialLocalStateRequest
          | FrozenLocalStateRequest
          | SharedLocalStateRequest
      ),
  ):
    if isinstance(hole, SharedLocalStateRequest):
      if state_sharing == "forbidden":
        raise ValueError(
            "Found a SharedLocalStateRequest for state variable"
            f" {hole.name} when state_sharing is set to 'forbidden'."
            " SharedLocalStateRequest must only be used when state_sharing is"
            " set to 'allowed' or 'unsafe'."
        )
      elif hole.name in initial_state_thunks:
        return HandledLocalStateRef(
            handler_id=handler_id,
            name=hole.name,
            was_explicitly_named=True,
            category=hole.category,
        )
      else:
        raise ValueError(
            "Found a SharedLocalStateRequest for state variable"
            f" {hole.name} before seeing a corresponding value."
            " SharedLocalStateRequest must appear after some other state"
            " request with the same name and an explicit value."
        )
    if isinstance(hole, InitialLocalStateRequest):
      thunk = hole.state_initializer
    else:
      thunk = lambda: hole.state
    if hole.name is None:
      auto_name = tree_util.pretty_keystr(keypath, body)
      ref = HandledLocalStateRef(
          handler_id=handler_id,
          name=auto_name,
          was_explicitly_named=False,
          category=hole.category,
      )
    else:
      ref = HandledLocalStateRef(
          handler_id=handler_id,
          name=hole.name,
          was_explicitly_named=True,
          category=hole.category,
      )
    if ref.name in initial_state_thunks:
      if state_sharing == "forbidden":
        raise ValueError(
            "Detected two local states with the same explicit name"
            f" {repr(ref.name)}, which is not allowed when state_sharing is"
            " set to 'forbidden'."
        )
      elif state_sharing == "allowed":
        if not (
            isinstance(hole, InitialLocalStateRequest)
            and initial_state_thunks[ref.name] is hole.state_initializer
        ):
          raise ValueError(
              "Detected two local states with the same explicit name"
              f" {repr(ref.name)} but different initializers! This is only"
              " allowed when state_sharing is set to 'unsafe'."
          )
      elif state_sharing == "unsafe":
        pass
      else:
        raise ValueError(f"Bad state sharing setting: {state_sharing}")
    initial_state_thunks[ref.name] = thunk
    return ref

  adjusted_body = (
      selectors.select(body)
      .at_instances_of(
          InitialLocalStateRequest
          | FrozenLocalStateRequest
          | SharedLocalStateRequest
      )
      .where(lambda req: category_predicate(req.category))
      .apply(_make_ref, with_keypath=True)
  )

  if lazy:
    states_out = lambda: {k: v() for k, v in initial_state_thunks.items()}
  else:
    states_out = {k: v() for k, v in initial_state_thunks.items()}

  handler = WithFunctionalLocalState(handler_id=handler_id, body=adjusted_body)
  return handler, states_out