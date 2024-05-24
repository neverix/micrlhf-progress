import os

from huggingface_hub import HfFileSystem
from safetensors.flax import load_file
import jax

sae_cache = {}
def get_sae(layer=20, revision=5, idx=0, model_dir="models", return_fname=False):
    key = (layer, revision)
    if key in sae_cache:
        return sae_cache[key]
    fs = HfFileSystem()
    weights = fs.glob(f"nev/phi-3-4k-saex-test/l{layer}-test-run-{revision}-*/*.safetensors")
    weight = sorted(weights)[idx]
    sparsity = float("-".join(weight.split("/")[2].split("-")[4:]))
    os.makedirs("models/sae", exist_ok=True)
    fname = f"{model_dir}/sae/{layer}-{revision}-{sparsity}.safetensors"
    w = "/".join(weight.split("/")[2:])
    os.system(f'wget -c "https://huggingface.co/nev/phi-3-4k-saex-test/resolve/main/{w}?download=true" -O "{fname}"')
    if return_fname:
        return fname
    sae_weights = load_file(fname)
    sae_cache[key] = sae_weights
    return sae_weights

def sae_encode(sae, vector):
    pre_relu = vector @ sae["W_enc"] + sae["b_enc"]
    post_relu = jax.nn.relu(pre_relu) * sae["scaling_factor"]
    decoded = post_relu @ sae["W_dec"] + sae["b_dec"]
    return pre_relu, post_relu, decoded
