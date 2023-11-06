from tokenizer import Tokenizer
from safetensors import safe_open
from llama2_model import LLaMA
import jax.numpy as jnp
import jax


tokenizer = Tokenizer("models/Llama-2-7b-hf/tokenizer.model")
input_ids = tokenizer.encode("Hello world!", bos=True, eos=False)
print(input_ids)
llama = LLaMA(jax.random.PRNGKey(32))
print(llama)
print(llama(jnp.asarray(input_ids)[None, :]))

with safe_open("/home/neverix/micrlhf/models/Llama-2-7b-hf/model-00001-of-00002.safetensors",
               framework="numpy", device="cpu") as f:
    for k, v in f.items():
        print(k, v.shape)
