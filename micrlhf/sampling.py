import dataclasses
import random
from collections import OrderedDict
from functools import partial
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import tiktoken
import transformers
from penzai import pz
from penzai.toolshed import jit_wrapper
from tqdm.auto import trange

from micrlhf.caching_llama import (FoldedLlamaKVCachingTransformer,
                                   LlamaKVCachingInputs, LlamaKVCachingState,
                                   LlamaKVCachingTransformer)
from micrlhf.llama import LlamaTransformer
from micrlhf.tokenizer import load_tokenizer


@jax.jit
def call_llama(llama_cached, inputs):
    return llama_cached(inputs)


@partial(jax.jit, donate_argnums=(1, 3), static_argnums=(4,))
def sample_logits(logits, tokens, cache, key, do_sample=False):
    if do_sample:
        key_sample, key = jax.random.split(key)
        choices = pz.nx.nmap(lambda l: jax.random.categorical(key_sample, l))(
            logits.untag("batch", "vocabulary")).tag("batch").untag("seq")[cache.cache_end_index - 1]
    else:
        choices = logits.untag("vocabulary").argmax().untag("seq")[cache.cache_end_index - 1]
    tokens = pz.nx.nmap(lambda t, c: t.at[cache.cache_end_index].set(c))(tokens.untag("seq"), choices).tag("seq")
    return choices, tokens, key


@partial(jax.jit, donate_argnums=(1, 2, 3, 4), static_argnums=(7,))
def sample_step(llama_cached, advanced, tokens, cache, key, base_mask, offsets, do_sample):
    batch_size = tokens.named_shape["batch"]
    max_seq_len = tokens.named_shape["seq"]
    inputs = LlamaKVCachingInputs(
        tokens=advanced[None].tag("seq"),
        positions=pz.nx.full({"batch": batch_size, "seq": 1}, cache.cache_end_index, jnp.int32) - offsets,
        attention_mask=((pz.nx.wrap(cache.cache_end_index) >= pz.nx.arange("kv_seq", max_seq_len, dtype=jnp.int32))
                        & (base_mask).untag("seq").tag("kv_seq"))[None].tag("seq"),
        sampling_state=cache
    )
    logits, cache = call_llama(llama_cached, inputs)
    advanced, tokens, key = sample_logits(logits, tokens, cache, key, do_sample)
    return advanced, tokens, cache, key

def sample(llama: Union[LlamaTransformer, Tuple[LlamaKVCachingTransformer, LlamaKVCachingState]],
           tokenizer: Union[tiktoken.Encoding | transformers.PreTrainedTokenizerBase],
            # TODO: multiple prompts and left padding
           prompt: Union[str, List[str]],
           batch_size: int = 1,
           max_seq_len: int = 64,
           pad_token_id: int = 128_020,
           do_sample: bool = False,
           return_model: bool = False,
           strip_padding: bool = True,
           return_only_completion: bool = False,
           seed: Optional[int] = None,
           verbose: bool = True,
           use_jit: bool = True,
           cache_kwargs: Optional[OrderedDict] = None,
           cfg_strength: float = 1.0,
           only_cache: bool = False,
           ):
    if getattr(tokenizer, "pad_token_id", None) is not None:
        pad_token_id = tokenizer.pad_token_id
    if isinstance(prompt, str):
        tokens = tokenizer.encode(prompt)
        initial_length = len(tokens)
        tokens = [tokens + [pad_token_id] * (max_seq_len - len(tokens))] * batch_size
        mask = [[1] * max_seq_len] * len(tokens)
        offsets = [0] * len(tokens)
    else:
        og_tokens = [tokenizer.encode(p) for p in prompt for _ in range(batch_size)]
        initial_length = max(map(len, og_tokens))
        offsets = [initial_length - len(t) for t in og_tokens]
        mask = [[0] * (initial_length - len(t)) + [1] * (max_seq_len - (initial_length - len(t))) for t in og_tokens]
        tokens = [[pad_token_id] * (initial_length - len(t)) + t for t in og_tokens]
        tokens = [t + [pad_token_id] * (max_seq_len - len(t)) for t in tokens]
        batch_size = len(tokens)

    assert initial_length < max_seq_len
    tokens = pz.nx.wrap(jnp.asarray(tokens, dtype=jnp.int32), "batch", "seq")
    base_mask = pz.nx.wrap(jnp.asarray(mask, dtype=jnp.bool_), "batch", "seq")
    offsets = pz.nx.wrap(jnp.asarray(offsets, dtype=jnp.int32), "batch")
    if isinstance(llama, tuple):
        llama_cached, cache = llama
    else:
        llama_cached, cache = LlamaKVCachingTransformer.from_uncached(llama,
                                                                      max_seq_len,
                                                                      {"batch": batch_size},
                                                                      **(cache_kwargs if cache_kwargs else {}))
    if return_model:
        llama_base = llama_cached
    if return_model:
        cache_base = cache
    if only_cache:
        return None, (llama_base, cache_base)

    # prefill
    base_inputs = llama_cached.inputs.from_basic_subsegments(tokens, cache)
    base_inputs = dataclasses.replace(base_inputs,
                                      positions=base_inputs.positions - offsets,
                                      attention_mask=base_inputs.attention_mask & base_mask.untag("seq").tag("kv_seq")
                                      )
    with jax.disable_jit(disable=not use_jit):
        logits, cache = call_llama(llama_cached, base_inputs)
        if cfg_strength != 1.0:
            def apply_cfg(logits):
                uncond, cond = jnp.split(logits, 2, axis=0)
                cond = uncond + (cond - uncond) * cfg_strength
                return jnp.concatenate((cond, cond), axis=0)
            logits = pz.nx.nmap(apply_cfg)(logits.untag("batch")).tag("batch")
        cache = dataclasses.replace(cache, cache_end_index=initial_length)

        key = jax.random.key(seed if seed is not None else random.randrange(0, 2**32))
        # generate
        advanced, tokens, key = sample_logits(logits, tokens, cache, key, do_sample=do_sample)

        for _ in (trange(max_seq_len - initial_length) if verbose else range(max_seq_len - initial_length)):
            advanced, tokens, cache, key = sample_step(llama_cached, advanced, tokens, cache, key,
                                                    base_mask, offsets, do_sample=do_sample)
            # bar.set_description(tokenizer.decode(tokens.untag("batch", "seq").data_array[0]))


    tokens = tokens.untag("batch", "seq").data_array
    if return_only_completion:
        tokens = tokens[:, initial_length:]
    elif strip_padding:
        tokens = [[a for a, b in zip(seq1, seq2) if b] for seq1, seq2 in zip(tokens, mask)]
    texts = [tokenizer.decode(sequence) for sequence in tokens]
    if return_model:
        return texts, (llama_base, cache_base)
    else:
        return texts, cache


def main(
    filename = "models/phi-3-16.gguf",
    # prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>Hi,",
    prompt = "<s><|system|>You are an assistant.</s><|user|>Hello!</s><|assistant|>Hi,",
):
    tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    llama = LlamaTransformer.from_pretrained(filename)
    print(sample(llama, tokenizer, prompt))


if __name__ == "__main__":
    main()
