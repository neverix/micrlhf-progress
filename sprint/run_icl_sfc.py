
import os
if "models" not in os.listdir("."):
    os.chdir("..")


import penzai
import jax
import jax_smi
import numpy as np

from tqdm.auto import tqdm
from penzai import pz
from tqdm.auto import trange
import jax.numpy as jnp
from collections import defaultdict


from sprint.icl_sfc_utils import Circuitizer


from micrlhf.llama import LlamaTransformer
llama = LlamaTransformer.from_pretrained("models/gemma-2b-it.gguf", from_type="gemma", load_eager=True, device_map="tpu:0")


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("alpindale/gemma-2b")
tokenizer.padding_side = "right"


from sprint.task_vector_utils import load_tasks, ICLRunner
tasks = load_tasks()


def check_if_single_token(token):
    return len(tokenizer.tokenize(token)) == 1



def main(task_name):
    task = tasks[task_name]
    task = {
        k:v for k,v in task.items() if check_if_single_token(k) and check_if_single_token(v)
    }

    pairs = list(task.items())

    batch_size = 8 
    n_shot=16
    max_seq_len = 128
    seed = 10

    prompt = "Follow the pattern:\n{}"

    if task_name.startswith("algo"):
        n_shot = 12

    runner = ICLRunner(task_name, pairs, batch_size=batch_size, n_shot=n_shot, max_seq_len=max_seq_len, seed=seed, prompt=prompt)


    layers = list(range(6, 17))
    circuitizer = Circuitizer(llama, tokenizer, runner, layers, prompt)


    import numpy as np
    thresholds = np.linspace(1e-7, 1e-5, 100)
    topks = [4, 6, 12, 16, 24, 32]


    ablated_metrics, n_nodes_counts = circuitizer.run_ablated_metrics(thresholds)


    target_metric = (max(ablated_metrics) - min(ablated_metrics)) * 0.95 + min(ablated_metrics)
    target_threshold = [threshold for threshold, metric in reversed(list(zip(thresholds, ablated_metrics))) if metric > target_metric][0]


    combined_ies = {}
    typed_ies = {
        "r": circuitizer.ie_resid,
        "a": circuitizer.ie_attn,
        "t": circuitizer.ie_transcoder,
    }

    for layer in tqdm(layers):
        for type in typed_ies:
            if layer in typed_ies[type]: 
                ies = typed_ies[type][layer]
                for mask in circuitizer.masks:
                    ies_mask = circuitizer.mask_average(ies, mask)
                    i = np.nonzero(np.abs(ies_mask) > target_threshold)[0]
                    w = ies_mask[i]

                    for idx, weight in zip(i.tolist(), w.tolist()):
                        combined_ies[(layer, mask, type, idx)] = weight

                    # w, i = jax.lax.top_k(ies_mask, 4)
                    # for idx, weight in zip(i.tolist(), w.tolist()):
                    #     combined_ies[(layer, mask, type, idx)] = weight


    combined_ies = [
        key + (weight,)
        for key, weight in combined_ies.items()
    ]


    typed_ies_error = {
        "er": circuitizer.ie_error_resid,
        "ea": circuitizer.ie_error_attn,
        "et": circuitizer.ie_error_transcoder,
    }

    for layer in tqdm(layers):
        for type in typed_ies_error:
            if layer in typed_ies_error[type]: 
                ies = typed_ies_error[type][layer]
                for mask in circuitizer.masks:
                    ies_mask = circuitizer.mask_average(ies, mask)
                    # print(ies_mask.tolist())
                    # raise
                    combined_ies.append((layer, mask, type, 0, ies_mask.tolist()))


    combined_ies = sorted(combined_ies, key=lambda x: -x[-1])


    important_feats_masks = {}
    for mask in circuitizer.masks:
        important_feats_masks[mask] = [
            (type, layer, feat) for layer, f_mask, type, feat, _ in combined_ies if f_mask == mask
            ]


    flat_feats = defaultdict(list)
    for k, v in important_feats_masks.items():
        for type, layer, feat in v:
            flat_feats[(k, type, layer)].append(feat)


    graph = []

    batch_size = 16
    k = 32
    for type, features in tqdm(sorted(flat_feats.items(), key=lambda x: (-x[0][-1], x[0][-2], x[0][-3]))):
        mask, feature_type, layer = type
        mask = jnp.array(list(circuitizer.masks.keys()).index(mask))
        for batch in trange(0, len(features), batch_size, postfix=str(type)):
            batch_features = features[batch:batch+batch_size]
            orig_length = len(batch_features)
            batch_features = batch_features + [0] * (batch_size - len(batch_features))
            feature_effectss = jax.vmap(lambda x: circuitizer.compute_feature_effects(feature_type, layer, x, mask, layer_window=1))(jnp.asarray(batch_features))
            # feature_effectss = circuitizer.compute_feature_effects(feature_type, layer, batch_features, mask, layer_window=1)
            top_effects = defaultdict(list)
            for key, featuress in feature_effectss.items():
                for elem, feature_effects in enumerate(featuress):
                    if elem >= orig_length:
                        continue
                    if feature_effects.ndim == 0:
                        top_effects[elem].append((float(feature_effects), key, 0))
                        continue
                    effects, indices = jax.lax.top_k(jnp.abs(feature_effects), k)
                    for i, e in zip(indices.tolist(), effects.tolist()):
                        top_effects[elem].append((e, key, i))
            for elem, effects in top_effects.items():
                effects.sort(reverse=True)
                edges = effects[:k]
                graph.extend([(weight,  key + (upstream_feature,), (type[1], type[2], type[0], batch_features[elem],) ) for weight, key, upstream_feature in edges])
            


    combined_ies = [
        (type, layer, mask, idx, weight) for layer, mask, type, idx, weight in combined_ies
    ] 


    sorted_graph = sorted(graph, reverse=True, key=lambda x: x[0])

    n_nodes = sum(map(len, important_feats_masks.values()))
    k_connections = 4
    weight_threshold = sorted_graph[n_nodes * k_connections][0]

    import json
    with open(f"micrlhf-progress/all-graph-{task_name}-fixed.json", 'w') as f:
        json.dump({"edges": graph, "nodes": combined_ies, "threshold": weight_threshold}, f)



from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()
    main(args.task)