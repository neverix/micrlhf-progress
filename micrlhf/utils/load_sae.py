import os
import shutil

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import HfFileSystem
from safetensors.flax import load_file, save_file

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

def get_jb_it_sae():
    key = "gemma_2b_jb_it_16k"
    if key not in sae_cache:
        os.makedirs("models/sae", exist_ok=True)
        fname = "models/sae/gemma_2b_it_blocks.12.hook_resid_post_16384.safetensors"
        os.system(f"wget -c 'https://huggingface.co/jbloom/Gemma-2b-IT-Residual-Stream-SAEs/resolve/main/gemma_2b_it_blocks.12.hook_resid_post_16384/sae_weights.safetensors?download=true' -O '{fname}'")
        sae_weights = load_file(fname)
        sae_cache[key] = sae_weights
    return sae_cache[key]

def get_nev_it_sae():
    key = "gemma_2b_nev_it_16k"
    if key not in sae_cache:
        os.makedirs("models/sae", exist_ok=True)
        fname = "models/sae/gemma_2b_it_nev_v0.safetensors"
        os.system(f"wget -c 'https://huggingface.co/nev/gemma-2b-saex-test/resolve/main/it-l12-residual-test-run-0-3.00E-05/sae_weights.safetensors?download=true' -O '{fname}'")
        sae_weights = load_file(fname)
        sae_cache[key] = sae_weights
    return sae_cache[key]

def get_nev_it_sae_suite(layer: int = 12, label = "residual", revision = 1, idx=0, model_dir="models"):
    key = f"gemma_2b_nev_it_{label}_{layer}"
    if key in sae_cache:
        return sae_cache[key]
    fs = HfFileSystem()
    weights = fs.glob(f"nev/gemma-2b-saex-test/it-l{layer}-{label}-test-run-{revision}-*/*.safetensors")
    weight = sorted(weights)[idx]
    sparsity = float("-".join(weight.split("/")[2].split("-")[6:]))
    os.makedirs("models/sae", exist_ok=True)
    fname = f"{model_dir}/sae/it-{layer}-{label}-{revision}-{sparsity}.safetensors"
    w = "/".join(weight.split("/")[2:])
    fname_16 = name_bf16(fname)
    if not os.path.exists(fname_16):
        with open(fname, "wb") as f:
            with fs.open(f"nev/gemma-2b-saex-test/{w}", "rb") as f2:
                f.write(f2.read())

        # os.system(f'wget -c "https://huggingface.co/nev/gemma-2b-saex-test/resolve/9e8944d087c755c4ead1f78ee0e9d8fd6b71187e/{w}?download=true" -O "{fname}"')
        convert_to_bf16(fname, fname_16)
    fname = fname_16
    sae_weights = load_file(fname)
    sae_cache[key] = sae_weights

    if "mean_norm" in sae_weights:
        norm_factor = (sae_weights["W_enc"].shape[0] ** 0.5) / sae_weights["mean_norm"]
        sae_weights["norm_factor"] = norm_factor
        if "tgt_mean_norm" in sae_weights:
            sae_weights["out_norm_factor"] = (sae_weights["W_enc"].shape[0] ** 0.5) / sae_weights["tgt_mean_norm"]
        else:
            sae_weights["out_norm_factor"] = norm_factor

    return sae_weights

def get_dm_res_sae(layer, load_65k=False, type="res", sparsity=None, nine_b=False):
    fs = HfFileSystem()
    b_key = "2b" if not nine_b else "9b"
    key = f"dm_gemma2_{b_key}_{type}_{layer}"
    if key in sae_cache:
        return sae_cache[key]

    width = 65 if load_65k else 16

    rep_url = f"google/gemma-scope-{b_key}-pt-{type}/layer_{layer}/width_{width}k/*"
    if type == "transcoder":
        rep_url = f"google/gemma-scope-{b_key}-pt-{type}/layer_{layer}/width_{width}k/*"

    # if type != "res":
    l0_versions = list(
        fs.glob(rep_url)
    )

    if sparsity is None:
        l0_versions = sorted(
            l0_versions, key=lambda x: abs(100 - int(x.split("average_l0_")[1]))
        )
        url = l0_versions[0]
    else:
        url = l0_versions[sparsity]
    url = url + "/params.npz"

    # else:
        # url = f"google/gemma-scope-2b-pt-{type}/layer_{layer}/width_{width}k/canonical/params.npz"

    os.makedirs("models/sae", exist_ok=True)
    if load_65k:
        fname = f"models/sae/dm_gemma2_2b_{type}_{layer}_{sparsity}_65k.npz"
    else:
        fname = f"models/sae/dm_gemma2_2b_{type}_{layer}_{sparsity}.npz"

    print(fname)

    if not os.path.exists(fname):
        with open(fname, "wb") as f:
            with fs.open(url, "rb") as f2:
                f.write(f2.read())
    state_dict = {}
    with np.load(fname) as data:
        for key in data.keys():
            state_dict_key = key
            if state_dict_key.startswith("w_"):
                state_dict_key = "W_" + state_dict_key[2:]
            state_dict[state_dict_key] = data[key]
    return state_dict

def get_nev_sae():
    key = "gemma_2b_nev_16k"
    if key not in sae_cache:
        os.makedirs("models/sae", exist_ok=True)
        fname = "models/sae/gemma_2b_nev_v0.safetensors"
        os.system(f"wget -c 'https://huggingface.co/nev/gemma-2b-saex-test/resolve/main/l12-test-run-5-3.00E-05/sae_weights.safetensors?download=true' -O '{fname}'")
        sae_weights = load_file(fname)
        sae_cache[key] = sae_weights
    return sae_cache[key]
    

def sae_encode(sae, vector, **kwargs):
    if "threshold" in sae:
        return sae_encode_threshold(sae, vector, **kwargs)
    if "s_gate" in sae:
        return sae_encode_gated(sae, vector, **kwargs)
    pre_relu = vector @ sae["W_enc"] + sae["b_enc"]
    post_relu = jax.nn.relu(pre_relu)
    if "scaling_factor" in sae:
        post_relu = post_relu * sae["scaling_factor"]
    decoded = post_relu @ sae["W_dec"] + sae["b_dec"]
    return pre_relu, post_relu, decoded

def resids_to_weights(vector, sae):
    inputs = vector

    if "norm_factor" in sae:
        inputs = inputs * sae["norm_factor"]

    pre_relu = inputs @ sae["W_enc"]
    pre_relu = pre_relu +sae["b_enc"]
    post_relu = jax.nn.relu(pre_relu)
    
    post_relu = (post_relu > 0) * jax.nn.relu((inputs @ sae["W_enc"]) * jax.nn.softplus(sae["s_gate"]) * sae["scaling_factor"] + sae["b_gate"])   

    return post_relu

def weights_to_resid(weights, sae):
    weights = jax.nn.relu(weights)

    recon = jnp.einsum("fv,...f->...v", sae["W_dec"], weights)

    recon = recon + sae["b_dec"]

    if "out_norm_factor" in sae:
        recon = recon / sae["out_norm_factor"]

    # recon = recon.astype('bfloat16')
    return recon.astype(weights.dtype)

def sae_encode_threshold(sae, vector, pre_relu=None, keep_features=None, ablate_to=0, post_relu=None):
    inputs = vector


    if post_relu is None:
        if pre_relu is None:
            pre_relu = inputs @ sae["W_enc"] + sae["b_enc"]

        post_relu = jax.nn.relu(pre_relu)
        post_relu = post_relu * (pre_relu > sae["threshold"])

    if keep_features is not None:
        post_relu = post_relu * keep_features + ablate_to * (1 - keep_features)

    recon = post_relu @ sae["W_dec"] + sae["b_dec"]

    return pre_relu, post_relu, recon

def sae_encode_gated(sae, vector, ablate_features=None, keep_features=None, pre_relu=None, post_relu=None, ablate_to=0):
    inputs = vector

    s = jax.nn.softplus(sae["s_gate"]) * sae["scaling_factor"]

    if post_relu is None:
        if pre_relu is None:
            if "norm_factor" in sae:
                inputs = inputs * sae["norm_factor"]

            pre_relu = inputs @ sae["W_enc"]
            pre_relu = pre_relu +sae["b_enc"]
            pre_relu = pre_relu * s

            if keep_features is not None:
                pre_relu = pre_relu * keep_features + ablate_to * (1 - keep_features)

        post_relu = jax.nn.relu(pre_relu)
    
    threshold = jnp.maximum(0, sae["b_gate"] - sae["b_enc"] * s)
    post_relu = (post_relu > threshold) * post_relu

    if keep_features is not None:
        post_relu = post_relu * keep_features + ablate_to * (1 - keep_features)
    
        # axes = tuple(range(post_relu.ndim - 1))
        
        # post_relu = jax.vmap(jax.vmap(lambda a, b: a.at[keep_features].set(b[keep_features]), in_axes=(0, 0), out_axes=0),
        #                      in_axes=(0, 0), out_axes=0)(jnp.zeros_like(post_relu), post_relu)

    if ablate_features is not None:
        post_relu = post_relu.at[ablate_features].set(0)

    recon = post_relu @ sae["W_dec"]

    recon = recon + sae["b_dec"]
    
    if "out_norm_factor" in sae:
        recon = recon / sae["out_norm_factor"]

    return pre_relu, post_relu, recon

def name_bf16(fname):
    return fname.replace(".safetensors", "_bf16.safetensors")

def convert_to_bf16(fname, fname_out):
    shutil.move(fname, fname_out)
    save_file({k: v.astype("bfloat16") for k, v in load_file(fname_out).items()}, fname_out)
