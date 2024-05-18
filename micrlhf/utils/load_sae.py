import os

from huggingface_hub import HfFileSystem
from safetensors.flax import load_file

sae_cache = {}
def get_sae(layer=20, revision=5, idx=0):
    key = (layer, revision)
    if key in sae_cache:
        return sae_cache[key]
    fs = HfFileSystem()
    weights = fs.glob(f"nev/phi-3-4k-saex-test/l{layer}-test-run-{revision}-*/*.safetensors")
    weight = sorted(weights)[idx]
    sparsity = float("-".join(weight.split("/")[2].split("-")[4:]))
    os.makedirs("models/sae", exist_ok=True)
    fname = f"models/sae/{layer}-{revision}-{sparsity}.safetensors"
    w = "/".join(weight.split("/")[2:])
    os.system(f'wget -c "https://huggingface.co/nev/phi-3-4k-saex-test/resolve/main/{w}?download=true" -O "{fname}"')
    sae_weights = load_file(fname)
    sae_cache[key] = sae_weights
    return sae_weights
