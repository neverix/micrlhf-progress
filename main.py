from tokenizer import Tokenizer
from llama2_model import LLaMA

from safetensors import safe_open
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import jax.numpy as jnp
import transformers
import numpy as np
import jax
import jmp


tokenizer = Tokenizer("models/Llama-2-7b-hf/tokenizer.model")
input_ids = tokenizer.encode("Hello world!", bos=True, eos=False)
print(input_ids)


num_devices = len(jax.devices())
mesh = Mesh(np.array(jax.devices()).reshape(2, 4), axis_names=("dp", "mp"))
policy = jmp.get_policy("p=bf16,c=bf16")

reference_llama = transformers.LlamaModel.from_pretrained("models/Llama-2-7b-hf")
llama = LLaMA(jax.random.PRNGKey(32), mesh, policy)
with safe_open(
    "/home/neverix/micrlhf/models/Llama-2-7b-hf/model-00001-of-00002.safetensors",
    framework="numpy",
    device="cpu",
) as f:
    for k in f.keys():
        print(f.get_slice(k).shape)
print(llama)
with mesh:
    input_ids = [
        tokenizer.encode(x, bos=True, eos=False)
        for x in ["Hello world", "This is a test"]
    ]
    input_ids = [x + [0] * (128 - len(x)) for x in input_ids]
    ids = jnp.asarray(input_ids)
    ids = jax.device_put(ids, NamedSharding(mesh, spec=PartitionSpec("dp", None)))
    print(llama(ids))
