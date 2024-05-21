import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from penzai import pz

from .llama import LlamaAttention, LlamaConfig, LlamaTransformer


@pz.pytree_dataclass
class LlamaKVCachingState(pz.Struct):
  cache_len: int = dataclasses.field(metadata={"pytree_node": False})
  batch_axes: dict[str, int] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  kv_caches: dict[str, Any]
  cache_end_index: int | jax.Array


# stolen from https://penzai.readthedocs.io/en/stable/notebooks/gemma_from_scratch.html
# could inherit from pz.nn.CachingAttention, but what's the fun in that?
@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class LlamaKVCachingAttention(pz.nn.KVCachingAttention):
  @classmethod
  def from_uncached(
      cls,
      original: LlamaAttention,
      cache_len: int,
      cached_axes: dict[str, int],
      cache_dtype: jax.typing.DTypeLike = jnp.float32,
  ) -> "LlamaKVCachingAttention":
    return super().from_uncached(
        original=original,
        sequence_axis="seq",
        cache_len=cache_len,
        cached_axes=cached_axes,
        cache_end_index_tag="cache_end_index",
        state_category="kv_cache",
        cache_dtype=cache_dtype,
    )


@pz.pytree_dataclass
class LlamaKVCachingInputs(pz.Struct):
  tokens: pz.nx.NamedArray
  positions: pz.nx.NamedArray
  attention_mask: pz.nx.NamedArray
  sampling_state: LlamaKVCachingState

  @classmethod
  def from_basic_subsegments(
      cls, tokens: pz.nx.NamedArray, sampling_state: LlamaKVCachingState
  ) -> "LlamaKVCachingInputs":
    """Constructs a simple input structure for a batch of unpadded samples.

    This can be used to process inputs that do not need advanced position or
    attention mask handling, and which just consist of ordinary sequences that
    are not packed together or padded. It augments the tokens with a standard
    position array and causal attention mask, adjusted by the current cache
    offset.

    Args:
      tokens: Subsquence of tokens, as an integer named array with a "seq" axis
        and possibly batch axes. When pre-filling, the "seq" axis can be the
        length of the prompt. When sampling, the "seq" instance will usually
        have length 1.
      sampling_state: Current sampling state, containing key-value caches.

    Returns:
      A full input structure containing the provided tokens, along with a simple
      incrementing position array and a causal mask, offset by the current
      sampling state.
    """
    seq = tokens.named_shape["seq"]
    offset = sampling_state.cache_end_index
    positions = pz.nx.arange("seq", seq) + offset
    # Query tokens can attend to keys/values if the query position is larger,
    # taking into account the cache offset.
    attention_mask = positions >= pz.nx.arange(
        "kv_seq", sampling_state.cache_len
    )
    return cls(
        tokens=tokens,
        positions=positions,
        attention_mask=attention_mask,
        sampling_state=sampling_state,
    )


@pz.pytree_dataclass
class LlamaKVCachingTransformer(pz.Layer):
  config: LlamaConfig = dataclasses.field(
      metadata={"pytree_node": False}
  )
  body: pz.LayerLike

  @pz.checked_layer_call
  def __call__(
      self, inputs: LlamaKVCachingInputs
  ) -> tuple[pz.nx.NamedArray, LlamaKVCachingState]:
    """Processes a new subsequence of tokens and adds them to the K/V cache.

    Args:
      inputs: Structure of input arguments, containing tokens, segment
        positions, an attention mask, and the current sampling state.

    Returns:
      A tuple ``(outputs, new_sampling_state)``, whre ``outputs`` is the final
      matrix of logits from the embedding decoding layer, which (in the normal
      configuration) will have axes "seq" and "vocabulary", and
      ``new_sampling_state`` is the updated sampling state with the updated
      key-value caches.
    """
    outs, kv_caches = self.body((
        (
            (inputs.tokens, inputs.positions, inputs.attention_mask),
            inputs.sampling_state.cache_end_index,
        ),
        inputs.sampling_state.kv_caches,
    ))
    return outs, LlamaKVCachingState(
        cache_len=inputs.sampling_state.cache_len,
        batch_axes=inputs.sampling_state.batch_axes,
        kv_caches=kv_caches,
        cache_end_index=(
            inputs.sampling_state.cache_end_index
            + inputs.tokens.named_shape["seq"]
        ),
    )

  def input_structure(self) -> pz.chk.StructureAnnotation:
    return LlamaKVCachingInputs(
        tokens=pz.chk.Wildcard("tokens"),
        positions=pz.chk.Wildcard("positions"),
        attention_mask=pz.chk.Wildcard("attention mask"),
        sampling_state=pz.chk.Wildcard("previous LlamaKVCachingState"),
    )

  def output_structure(self) -> pz.chk.StructureAnnotation:
    return (
        pz.chk.Wildcard("unnormalized logits"),
        pz.chk.Wildcard("updated LlamaKVCachingState"),
    )

  @classmethod
  def from_uncached(
      cls,
      uncached: LlamaTransformer,
      cache_len: int,
      batch_axes: dict[str, int],
  ) -> tuple["LlamaKVCachingTransformer", LlamaKVCachingState]:
    """Transforms a `LlamaTransformer` into cached sampling mode.

    This constructor hot-swaps all `model_core.GemmaAttention` layers in the
    original model to enable key-value caching, then installs new handlers to
    update their states appropriately. Note that any modifications to the
    uncached model will persist in the decoding mode.

    Args:
      uncached: The original `LlamaTransformer` model.
      cache_len: Maximum sequence length for the key/value caches.
      batch_axes: Names and sizes for the batch axes that will be used for
        sampling. Required for initializing the key/value caches.

    Returns:
      Tuple ``(sampler_model, initial_sampling_state)``, where ``sampler_model``
      is a `LlamaKVCachingTransformer`, and ``initial_sampling_state`` holds the
      initial empty key/value caches.
    """
    cached_axes = {
        **batch_axes,
        "projection": uncached.config.projection_dim,
    }
    cached_axes["kv_heads"] = uncached.config.num_key_value_heads
    caching_body = (
        pz.select(uncached.body)
        .at_instances_of(LlamaAttention)
        .apply(
            lambda attn: LlamaKVCachingAttention.from_uncached(
                attn,
                cache_len=cache_len,
                cached_axes=cached_axes,
                cache_dtype=uncached.config.activation_dtype,
            )
        )
    )
    handled_body, initial_state = pz.de.handle_local_states(
        pz.de.WithSideInputsFromInputTuple.handling(
            caching_body, tags=["cache_end_index"]
        ),
        category="kv_cache",
    )
    inference_model = cls(config=uncached.config, body=handled_body)
    sampling_state = LlamaKVCachingState(
        cache_len=cache_len,
        batch_axes=batch_axes,
        kv_caches=initial_state,
        cache_end_index=0,
    )
    return inference_model, sampling_state

  @property
  def inputs(self):
      return LlamaKVCachingInputs
