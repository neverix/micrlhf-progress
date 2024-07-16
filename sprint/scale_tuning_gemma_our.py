# %%
import os
if "models" not in os.listdir("."):
    os.chdir("..")

import penzai
from penzai import pz

# %%
from micrlhf.llama import LlamaTransformer
llama = LlamaTransformer.from_pretrained("models/gemma-2b-it.gguf", from_type="gemma", load_eager=True)

# %%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("alpindale/gemma-2b")
tokenizer.padding_side = "right"
# %%
from micrlhf.utils.activation_manipulation import ActivationReplacement
from micrlhf.utils.activation_manipulation import replace_activation, collect_activations
from micrlhf.sampling import sample
from penzai.toolshed.jit_wrapper import Jitted
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import random


FEAT_BATCH = 6
OPT_BATCH = 6
PICK_BATCH = 64
MIN_SCALE = 0
MAX_SCALE = 200
REP_LAYER = 2
MAX_LENGTH = 64
PROBE_LAYER = 16
CFG = 1.0
# PROMPT_TEMPLATE = '<start_of_turn>user\nRepeat "X" two times exactly as it is written.<end_of_turn>\n<start_of_turn>model\n1. "X"\n2. "'
PROMPT_TEMPLATE = '<start_of_turn>user\nRepeat "X" four times exactly as it is written.<end_of_turn>\n<start_of_turn>model\n1. "X"\n2. "X"\n3. "X"\n4. "'
# PROMPT_TEMPLATE = '<start_of_turn>user\nWhat is the meaning of the word "X"?<end_of_turn>\n<start_of_turn>model\nThe meaning of the word "X" is "'

POSITIONS = tuple(i for i, a in enumerate(tokenizer.encode(PROMPT_TEMPLATE)) if tokenizer.decode([a]) == "X")

embeds = llama.select().at_instances_of(pz.nn.EmbeddingLookup).get_sequence()[0].table.embeddings.value.unwrap("vocabulary", "embedding")
embed_mean = embeds.mean(axis=0)
embed_vector = embed_mean / jnp.linalg.norm(embed_mean)
tiled_embed = jnp.tile(embed_vector, (PICK_BATCH * OPT_BATCH, 1))
act_rep_base = Jitted(collect_activations(replace_activation(llama, tiled_embed, POSITIONS, layer=REP_LAYER)))


def benchmark_vector(vector, tokens, positions, replacement_layer):
    assert replacement_layer == REP_LAYER
    dumb = False
    if vector.ndim == 1:
        dumb = True
        vector = jnp.tile(vector[None, :], (PICK_BATCH * OPT_BATCH, 1))
    assert vector.shape == tiled_embed.shape
    assert positions == POSITIONS
    # act_rep = collect_activations(replace_activation(llama, vector, positions, layer=replacement_layer))
    act_rep = act_rep_base.select().at_instances_of(ActivationReplacement).apply(lambda x: ActivationReplacement.replace_vector(x, vector))
    logits, residuals = act_rep(tokens)
    result = logits, [r.value for r in residuals]
    if dumb:
        return result[0].untag("batch")[:1].tag("batch"), [r.untag("batch")[:1].tag("batch") for r in result[1]]
    return result


def tokens_to_inputs(tokens):
    token_array = jnp.asarray(tokens)
    token_array = jax.device_put(token_array, jax.sharding.NamedSharding(llama.mesh, jax.sharding.PartitionSpec("dp", "sp")))
    token_array = pz.nx.wrap(token_array, "batch", "seq").untag("batch").tag("batch")

    inputs = llama.inputs.from_basic_segments(token_array)
    return inputs


def pick_scale(features, batch_size=PICK_BATCH, min_scale=MIN_SCALE, max_scale=MAX_SCALE, layer=REP_LAYER):
    scales = np.linspace(min_scale, max_scale, batch_size)
    vector = [
        feature[None, :] * jnp.array(scales)[:, None]
        for feature in features
    ]
    vector = jnp.concatenate(vector, axis=0)

    text = [PROMPT_TEMPLATE for _ in range(batch_size * OPT_BATCH)]
    tokenized = tokenizer(text, return_tensors="np", padding="max_length", max_length=64, truncation=True)
    tokens = tokenized["input_ids"]
    inputs = tokens_to_inputs(tokens)
    
    logits, residuals = benchmark_vector(vector, inputs, POSITIONS, layer)
    
    # rand_vector = jax.random.normal(key=jax.random.key(random.randint(-10, 10)), shape=vector.shape)
    # rand_vector = rand_vector / jnp.linalg.norm(rand_vector, axis=-1, keepdims=True)
    # logits_rand, _ = benchmark_vector(rand_vector, inputs, positions, layer)
    
    logits_mean, _ = benchmark_vector(embed_vector, inputs, POSITIONS, layer)

    logits = logits.unwrap("batch", "seq", "vocabulary")
    entropies = -jnp.sum(jax.nn.log_softmax(logits) * jnp.exp(jax.nn.log_softmax(logits)), axis=-1)
    entropy_first = entropies[:, -1]

    entropy_first = [
        entropy_first[i * batch_size: (i + 1) * batch_size] - entropy_first[i * batch_size]
        for i in range(OPT_BATCH)
    ]
    

    resid_cos_features = [[] for _ in range(OPT_BATCH)]   
    for i in range(OPT_BATCH):
        for residual in residuals:
            resid = residual.unwrap("batch", "seq", "embedding")[i * batch_size: (i + 1) * batch_size, -1]
            resid_cos_feature = resid @ features[i] / jnp.linalg.norm(resid) / jnp.linalg.norm(features[i])
            resid_cos_feature = resid_cos_feature - resid_cos_feature[0]
            resid_cos_features[i].append(resid_cos_feature)
    
    crossents = []
    # for baseline in (logits_rand, logits_mean):
    for baseline in (logits_mean,):
        baseline = baseline.unwrap("batch", "seq", "vocabulary")
        # baseline_probs = jax.nn.softmax(baseline)
        # crossents.append(-jnp.sum(jax.nn.log_softmax(logits) * baseline_probs, axis=-1)[:, -1])
        crossents.append(-jnp.sum(jax.nn.log_softmax(baseline) * jax.nn.softmax(logits), axis=-1)[:, -1])
    
    crossents = [
        c[i * batch_size: (i + 1) * batch_size]
        for c in crossents
        for i in range(OPT_BATCH)
    ]

    return scales, entropy_first, resid_cos_features, crossents


def generate_explanations(feature=None, batch_size=32, min_scale=MIN_SCALE, max_scale=MAX_SCALE, layer=REP_LAYER, cfg=CFG, for_cache=False, cached=None, cache_batch=FEAT_BATCH):
    if feature is None:
        feature = jnp.zeros(2048)
        if cache_batch > 1:
            feature = jnp.tile(feature[None], (cache_batch, 1))
    scales = np.linspace(min_scale, max_scale, batch_size, dtype=np.float32)
    if feature.ndim == 1:
        vector = feature[None, :] * jnp.array(scales)[:, None]
    else:
        vector = feature[:, None, :] * jnp.array(scales)[None, :, None]
        vector = vector.reshape(-1, vector.shape[-1])
    if cfg != 1.0:
        vector = jnp.concatenate([embed_vector[None, :] * jnp.array(scales)[:, None], vector], axis=0)
    if cached is not None:
        act_rep = cached[0].select().at_instances_of(ActivationReplacement).apply(lambda x: ActivationReplacement.replace_vector(x, vector)), cached[1]
        # act_rep = cached
    else:
        act_rep = replace_activation(llama, vector, POSITIONS, layer=layer)
    completions, model = sample(act_rep, tokenizer,
                         PROMPT_TEMPLATE, batch_size=(batch_size if cfg == 1.0 else batch_size * 2) * (1 if feature.ndim == 1 else feature.shape[0]),
                         do_sample=True, max_seq_len=MAX_LENGTH,
                         return_only_completion=True,
                         verbose=False, cfg_strength=cfg,
                         return_model=for_cache, only_cache=for_cache)
    if for_cache:
        return model
    if feature.ndim == 1:
        return list(zip(np.concatenate((scales, scales)), completions))[batch_size if cfg != 1.0 else 0:]
    else:
        completions = np.array(completions).reshape(-1, batch_size if cfg == 1.0 else 2 * batch_size)
        explanations = []
        for c in completions:
            explanations.append(list(zip(np.concatenate((scales, scales)), c))[batch_size if cfg != 1.0 else 0:])
        return explanations


cached_model = generate_explanations(for_cache=True)
from tqdm.auto import tqdm
import json
from itertools import batched

def main(generations_filename, w_dec):
    n_features = w_dec.shape[0]
    if os.path.exists(generations_filename):
        with open(generations_filename, "r") as f:
            generations = [json.loads(l) for l in f]
            generations = {int(g["feature"]): g for g in generations}
    else:
        generations = {}
    random.seed(9)
    feat_batch = FEAT_BATCH
    data_points = []

    try:
        for batch in tqdm(list(batched(range(n_features), OPT_BATCH))):
            batch = [b for b in batch if b not in generations]
            if not batch:
                continue
            scales, entropy, selfsims, crossents = pick_scale([w_dec[b] for b in batch])
            for i, f in enumerate(batch):
                data_point = dict(
                    feature=f,
                    explanation="",
                    settings=dict(
                        min_scale=MIN_SCALE,
                        max_scale=MAX_SCALE,
                        rep_layer=REP_LAYER,
                        probe_layer=PROBE_LAYER,
                        cfg=CFG,
                    ),
                    scale_tuning=dict(
                        scales=scales.tolist(),
                        entropy=entropy[i].tolist(),
                        selfsims=[s.tolist() for s in selfsims[i]],
                        crossents=[c.tolist() for c in crossents[i]],
                    )
                )
                data_points.append(data_point)
            if len(data_points) < feat_batch:
                continue
            explanations = generate_explanations(jnp.stack([w_dec[p["feature"]] for p in data_points]), cached=cached_model)
            for p, e in zip(data_points, explanations):
                data_point = dict(
                    **p,
                    generations=[(float(a), b) for a, b in e],
                )
                generations[p["feature"]] = data_point

                with open(generations_filename, "a") as f:
                    f.write(json.dumps(data_point) + "\n")
            data_points = []
    except KeyboardInterrupt:
        pass

from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("save_path", type=str)
parser.add_argument("layer", type=int)
parser.add_argument("label", type=str)

from micrlhf.utils.load_sae import get_nev_it_sae_suite

if __name__ == "__main__":
    args = parser.parse_args()

    sae = get_nev_it_sae_suite(layer=args.layer, label=args.label)
    w_dec = sae["W_dec"]
    main(args.save_path, w_dec)

