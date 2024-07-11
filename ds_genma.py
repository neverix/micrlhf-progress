#!/usr/bin/env python
# coding: utf-8
N_GEN = 1
import penzai
from penzai import pz
filename = "models/phi-"
from micrlhf.llama import LlamaTransformer
llama = LlamaTransformer.from_pretrained("models/gemma-2b-it.gguf", from_type="gemma", load_eager=True)
from micrlhf.sampling import sample, trange, jnp, load_tokenizer, jit_wrapper
from transformers import AutoTokenizer
import jax
tokenizer = AutoTokenizer.from_pretrained("NousResearch/gemma-2b-it-tokenizer")
prompt = "<bos><start_of_turn>user\n"
batch_size: int = 128
max_seq_len: int = 256
pad_token_id: int = 0
from micrlhf.sampling import *
llama_cached, base_cache = LlamaKVCachingTransformer.from_uncached(llama, max_seq_len, {"batch": batch_size})
llama_cached_jitted = jit_wrapper.Jitted(llama_cached)
tokens = tokenizer.encode(prompt)
initial_length = len(tokens)
tokens = [tokens + [pad_token_id] * (max_seq_len - len(tokens))]
tokens = jnp.asarray(tokens, dtype=jnp.int32)
tokens = pz.nx.NamedArray(OrderedDict(batch=1, seq=max_seq_len), tokens)
tokens = tokens.untag("batch").repeat(batch_size).tag("batch")
# prefill
base_inputs = LlamaKVCachingInputs.from_basic_subsegments(tokens, base_cache)
base_mask = tokens != pad_token_id
base_inputs = dataclasses.replace(base_inputs,
                                    attention_mask=base_inputs.attention_mask & base_mask.untag("seq").tag("kv_seq")
                                    )
logits, cache = llama_cached_jitted(base_inputs)

import numpy as np
import random

@jax.jit
def sample(logits, tokens, cache, key):
    logits = pz.nx.nmap(lambda l, t: l - jax.lax.scan(lambda c, x: (c + jax.nn.one_hot(x, l.shape[-1], dtype=l.dtype), None), jnp.zeros_like(l), t)[0] # jax.nn.one_hot(t, l.shape[-1], dtype=l.dtype).sum(0)
                        )(logits.untag("seq", "vocabulary"),
                          tokens.untag("seq")).tag("seq", "vocabulary")
    choices = pz.nx.nmap(lambda l: jax.random.categorical(key, l))(
        logits.untag("batch", "vocabulary")).tag("batch").untag("seq")[cache.cache_end_index - 1]
    tokens = pz.nx.nmap(lambda t, c: t.at[cache.cache_end_index].set(c))(tokens.untag("seq"), choices).tag("seq")
    return choices, tokens, key

@partial(jax.jit, donate_argnums=(1, 2, 3, 4))
def sample_step(llama_cached, advanced, tokens, cache, key):
    inputs = LlamaKVCachingInputs(
        tokens=advanced[None].tag("seq"),
        positions=pz.nx.full({"batch": batch_size, "seq": 1}, cache.cache_end_index, jnp.int32),
        attention_mask=((pz.nx.wrap(cache.cache_end_index) >= pz.nx.arange("kv_seq", max_seq_len, dtype=jnp.int32))
                        & (base_mask | (pz.nx.arange("seq", max_seq_len, dtype=jnp.int32) >= initial_length)
                            ).untag("seq").tag("kv_seq"))[None].tag("seq"),
        sampling_state=cache
    )
    logits, cache = llama_cached(inputs)
    advanced, tokens, key = sample(logits, tokens, cache, key)
    return advanced, tokens, cache, key

def get_texts(cache=cache, tokens=tokens):
    cache = dataclasses.replace(cache, cache_end_index=initial_length)
    cache = dataclasses.replace(cache, kv_caches=jax.tree.map(lambda x: jnp.copy(x), cache.kv_caches))
    tokens = pz.nx.nmap(lambda t: jnp.copy(t))(tokens)
    key = jax.random.key(random.randint(0, 2**32))
    advanced, tokens, key = sample(logits, tokens, cache, key)
    for _ in range(max_seq_len):
        advanced, tokens, cache, key = sample_step(llama_cached_jitted, advanced, tokens, cache, key)

    tokens = np.array(tokens.untag("batch", "seq").data_array)
    return [tokenizer.decode(sequence[1:]) for sequence in tokens]

def gen(n=N_GEN):
# def gen(n=10_000):
    for _ in trange(n):
        for t in get_texts():
            yield {"text": t}

random.seed(19)
dataset = list(gen())
print(dataset)
4/0

import datasets
ds_gen = datasets.Dataset.from_list(dataset)

ds_gen.push_to_hub("nev/generated-gemma-format-text-0", )
