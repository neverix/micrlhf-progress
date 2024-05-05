import dataclasses
from collections import OrderedDict
from functools import partial

import jax
import jax.numpy as jnp
import tiktoken
import transformers
from penzai import pz
from penzai.toolshed import jit_wrapper
from tqdm.auto import trange

from micrlhf.caching_llama import LlamaKVCachingInputs, LlamaKVCachingTransformer
from micrlhf.llama import LlamaTransformer
from micrlhf.tokenizer import load_tokenizer


def sample(llama: LlamaTransformer, tokenizer : tiktoken.Encoding | transformers.PreTrainedTokenizerBase,
            # TODO: multiple prompts and left padding
           prompt: str,
           batch_size: int = 4,
           max_seq_len: int = 64,
           pad_token_id: int = 128_020):
    tokens = tokenizer.encode(prompt)
    initial_length = len(tokens)
    tokens = [tokens + [pad_token_id] * (max_seq_len - len(tokens))]
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    tokens = pz.nx.NamedArray(OrderedDict(batch=1, seq=max_seq_len), tokens)
    tokens = tokens.untag("batch").repeat(batch_size).tag("batch")
    llama_cached, cache = LlamaKVCachingTransformer.from_uncached(llama, max_seq_len, {"batch": batch_size})
    llama_cached = jit_wrapper.Jitted(llama_cached)

    def sample(logits, tokens, cache, key):
        choices = logits.untag("vocabulary").argmax().untag("seq")[cache.cache_end_index - 1]
        tokens = pz.nx.nmap(lambda t, c: t.at[cache.cache_end_index].set(c))(tokens.untag("seq"), choices).tag("seq")
        return choices, tokens, key

    # prefill
    base_inputs = LlamaKVCachingInputs.from_basic_subsegments(tokens, cache)
    base_mask = tokens != pad_token_id
    base_inputs = dataclasses.replace(base_inputs,
                                      attention_mask=base_inputs.attention_mask & base_mask.untag("seq").tag("kv_seq")
                                      )
    logits, cache = llama_cached(base_inputs)
    cache = dataclasses.replace(cache, cache_end_index=initial_length)

    key = jax.random.key(0)
    # generate
    advanced, tokens, key = sample(logits, tokens, cache, key)
    
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

    for _ in (bar := trange(max_seq_len)):
       advanced, tokens, cache, key = sample_step(llama_cached, advanced, tokens, cache, key)
        # bar.set_description(tokenizer.decode(tokens.untag("batch", "seq").data_array[0]))

    return [tokenizer.decode(sequence) for sequence in tokens.untag("batch", "seq").data_array]


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
