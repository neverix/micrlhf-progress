#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
import penzai
from penzai import pz
# pz.ts.register_as_default()
# pz.ts.register_autovisualize_magic()
# pz.enable_interactive_context()


# In[2]:


# filename = "models/Meta-Llama-3-8B-Instruct.Q8_0.gguf"
filename = "models/phi-3-16.gguf"
# filename = "models/tinyllama-1.1b-q8_0.gguf"


# In[3]:


from micrlhf.llama import LlamaTransformer
llama = LlamaTransformer.from_pretrained(filename, device_map="auto")


# In[4]:


from micrlhf.sampling import sample, trange, jnp, load_tokenizer, jit_wrapper
from transformers import AutoTokenizer
import jax
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
prompt = "<|user|>"
batch_size: int = 128
max_seq_len: int = 256
pad_token_id: int = 128_020
tokens = tokenizer.encode(prompt)
token_array = jnp.asarray(tokens).reshape((1, -1))
token_array = jnp.repeat(token_array, batch_size, axis=0)
token_array = jax.device_put(token_array, jax.sharding.NamedSharding(llama.mesh, jax.sharding.PartitionSpec("dp", "sp")))
token_array = pz.nx.wrap(token_array, "batch", "seq").untag("batch").tag("batch")
inputs = llama.inputs.from_basic_segments(token_array)


# In[5]:


from micrlhf.sampling import *
from micrlhf.scan import sequential_to_scan
llama_cached, base_cache = LlamaKVCachingTransformer.from_uncached(llama, max_seq_len, {"batch": batch_size})
llama_cached_jitted = jit_wrapper.Jitted(llama_cached)


# In[6]:


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


# In[15]:


import numpy as np
import random


def sample(logits, tokens, cache, key):
    choices = pz.nx.nmap(lambda l: jax.random.categorical(key, l))(
        logits.untag("batch", "vocabulary")).tag("batch").untag("seq")[cache.cache_end_index - 1]
    tokens = pz.nx.nmap(lambda t, c: t.at[cache.cache_end_index].set(c))(tokens.untag("seq"), choices).tag("seq")
    return choices, tokens, key


@partial(jax.jit, donate_argnums=(1, 2, 3, 4))
def sample_step(llama_cached, advanced, tokens, cache, key):
    inputs = LlamaKVCachingInputs(
        tokens=advanced[None].tag("seq"),
        positions=pz.nx.full({"batch": batch_size, "seq": 1}, cache.cache_end_index, jnp.int32),
        attention_mask=((cache.cache_end_index >= pz.nx.arange("kv_seq", max_seq_len, dtype=jnp.int32))
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


# In[33]:


def gen(n=2_000):
    for _ in trange(n):
        for t in get_texts():
            yield {"text": t}


# In[34]:


dataset = list(gen())


# In[ ]:


import datasets
ds_phi = datasets.Dataset.from_list(dataset)


# In[ ]:


ds_phi.push_to_hub("nev/generated-phi-format-text")


# In[ ]:




