import os
from functools import partial

import tiktoken
from tiktoken.load import load_tiktoken_bpe

from .gguf import GGUFReader


def load_tokenizer(gguf_path: os.PathLike):
    gguf = GGUFReader(gguf_path)
    tokenizer_keys = ["tokens", "scores", "token_type", "merges", "bos_token_id", "eos_token_id"]
    tokenizer_data = {k: gguf.metadata.get(f"tokenizer.ggml.{k}") for k in tokenizer_keys}
    special_id = tokenizer_data["bos_token_id"]
    if special_id <= 1:  # hack for phi
        end_id = tokenizer_data["eos_token_id"]
        if end_id <= 2:  # hack for tinyllama
            normal, special = tokenizer_data["tokens"][end_id+1:], tokenizer_data["tokens"][:end_id+1]
        else:
            normal, special = (
                tokenizer_data["tokens"][special_id+1:end_id],
                tokenizer_data["tokens"][:special_id+1] + tokenizer_data["tokens"][end_id:])
    else:
        normal, special = tokenizer_data["tokens"][:special_id], tokenizer_data["tokens"][special_id:]
    tokenizer = tiktoken.Encoding(
        name="tokenizer",
        pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        mergeable_ranks={t.replace("Ġ", " ").encode("utf-8"): i for i, t in enumerate(normal)},
        special_tokens={t: i + len(normal) for i, t in enumerate(special)},
    )
    tokenizer.encode = partial(tokenizer.encode, allowed_special=set(special))
    return tokenizer
