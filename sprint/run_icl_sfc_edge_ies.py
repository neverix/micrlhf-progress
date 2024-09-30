import os
import jax
if "models" not in os.listdir("."):
    os.chdir("..")

import penzai
from penzai import pz


# %%
from sprint.icl_sfc_utils import Circuitizer

# %%
from micrlhf.llama import LlamaTransformer
llama = LlamaTransformer.from_pretrained("models/gemma-2b-it.gguf", from_type="gemma", load_eager=True, device_map="tpu:0")

# %%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("alpindale/gemma-2b")
tokenizer.padding_side = "right"

# %%
from sprint.task_vector_utils import load_tasks, ICLRunner
tasks = load_tasks()


def main(task_name):    
    task = tasks[task_name]

    pairs = list(task.items())

    batch_size = 8 
    n_shot=16
    max_seq_len = 128
    seed = 10

    prompt = "Follow the pattern:\n{}"

    runner = ICLRunner(task_name, pairs, batch_size=batch_size, n_shot=n_shot, max_seq_len=max_seq_len, seed=seed, prompt=prompt, use_same_examples=True)

    layers = list(range(8, 16))
    circuitizer = Circuitizer(llama, tokenizer, runner, layers, prompt)

    layers = list(range(10, 15))
    mean_ablate = False

    orig_metric = circuitizer.ablated_metric(llama).tolist()
    zero_metric = circuitizer.run_ablated_metrics([100000], mean_ablate=mean_ablate, layers=layers)[0][0]

    print(orig_metric, zero_metric)

    # %%
    import numpy as np
    # thresholds = np.linspace(0, 1e-4, 100)
    # thresholds = np.linspace(1.4 * 1e-4, 1.45 * 1e-4, 200)
    thresholds = np.logspace(-4, -1, 150)
    topks = [4, 6, 12, 16, 24, 32]

    inverse = False
    do_abs = False
    average_over_positions = True


    ablated_metrics, n_nodes_counts = circuitizer.run_ablated_metrics(thresholds, inverse=inverse, 
                                                                    do_abs=do_abs, mean_ablate=mean_ablate, 
                                                                    average_over_positions=average_over_positions,
                                                                    token_prefix=None, layers=layers)

    faithfullness = np.array(ablated_metrics)
    faithfullness = (faithfullness - zero_metric) / (orig_metric - zero_metric)



    target_faithfullness = 0.6

    target_threshold = [threshold for threshold, metric in reversed(list(zip(thresholds, faithfullness))) if metric > target_faithfullness][0]

    target_threshold

    # %%
    from tqdm.auto import tqdm

    selected_threshold = target_threshold


    ablation_masks = {}

    for layer in tqdm(layers):
        mask_attn_out, _ = circuitizer.mask_ie(circuitizer.ie_attn[layer], selected_threshold, None, do_abs=do_abs, average_over_positions=average_over_positions, inverse=inverse)
        mask_resid, _ = circuitizer.mask_ie(circuitizer.ie_resid[layer], selected_threshold, None, do_abs=do_abs, average_over_positions=average_over_positions, inverse=inverse)

        # print(mask_resid["arrow"].shape)

        # break

        try:
            mask_transcoder, _ = circuitizer.mask_ie(circuitizer.ie_transcoder[layer], selected_threshold, None, do_abs=do_abs, average_over_positions=average_over_positions, inverse=inverse)
        except KeyError:
            mask_transcoder = None

        ablation_masks[layer] = {
            "attn_out": mask_attn_out,
            "resid": mask_resid,
            "transcoder": mask_transcoder
        }

    # %%
    circuit_nodes = []
    n_nodes = 0

    for layer, masks in ablation_masks.items():
        for mask_type, mask in masks.items():
            if mask is not None:
                for token_type, mask in mask.items():
                        n_nodes += mask.sum()
                        
                        node_ids = np.where(mask)

                        if len(node_ids) ==2:
                            for pos, feat in zip(*node_ids):
                                circuit_nodes.append((layer, mask_type, token_type, feat, pos))
                        else:
                            for feat in node_ids[0]:
                                circuit_nodes.append((layer, mask_type, token_type, feat, None))
                        

    n_nodes

    # %%
    typed_ies = {
        "r": circuitizer.ie_resid,
        "a": circuitizer.ie_attn,
        "t": circuitizer.ie_transcoder,
    }

    circuit_nodes_with_ies = []

    for node in circuit_nodes:
        layer, sae_type, token_type, node_id, pos = node
        ies = typed_ies[sae_type[0]][layer]

        if average_over_positions:
            masked_ies = circuitizer.mask_average(ies, token_type, average_over_positions=True)
            circuit_nodes_with_ies.append((*node, masked_ies[node_id].tolist()))
        else:
            masked_ies = circuitizer.mask_average(ies, token_type, average_over_positions=False)
            circuit_nodes_with_ies.append((*node, masked_ies[pos, node_id].tolist()))

    circuit_nodes_with_ies = sorted(circuit_nodes_with_ies, key=lambda x: x[-1], reverse=True)

    # %%
    circuit_nodes_with_ies[:10]

    # %%
    from tqdm.auto import tqdm
    import numpy as np

    combined_ies = {}

    if average_over_positions:
        for node in circuit_nodes_with_ies:
            layer, type, mask, idx, pos, ie = node
            combined_ies[(layer, mask, type[0], idx)] = ie

    else:
        for node in circuit_nodes_with_ies:
            layer, type, mask, idx, pos, ie = node
            combined_ies[(layer, mask, type[0], idx, pos)] = ie

    # %%
    combined_ies = [
        key + (weight,)
        for key, weight in combined_ies.items()
    ]

    # %%
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
                    ies_mask = circuitizer.mask_average(ies, mask, average_over_positions=average_over_positions)
                    # print(ies_mask.tolist())
                    # raise

                    if average_over_positions:
                        combined_ies.append((layer, mask, type, 0, ies_mask.tolist()))

                    else:
                        for pos, ie in enumerate(ies_mask):
                            if ie > selected_threshold:
                                combined_ies.append((layer, mask, type, 0, pos, ie))

    # %%
    combined_ies = sorted(combined_ies, key=lambda x: -x[-1])

    # %%

    from collections import defaultdict
    circuit_node_dict = defaultdict(list)

    if average_over_positions:
        for node in combined_ies:
            layer, mask, type, idx, weight = node
            circuit_node_dict[(type, layer, mask)].append(idx)

        circuit_node_dict = {
            k: np.array(v) for k,v in circuit_node_dict.items()
        }
    else:
        for node in combined_ies:
            layer, mask, type, idx, pos, weight = node
            circuit_node_dict[(type, layer, mask)].append((pos, idx))

        circuit_node_dict = {
            k: np.array(v) for k,v in circuit_node_dict.items()
        }

    # %%
    import jax.numpy as jnp
    from tqdm.auto import trange

    if average_over_positions:
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
        # k = 32
        for type, features in tqdm(sorted(flat_feats.items(), key=lambda x: (-x[0][-1], x[0][-2], x[0][-3]))):
            mask, feature_type, layer = type
            mask = jnp.array(list(circuitizer.masks.keys()).index(mask))
            for batch in trange(0, len(features), batch_size, postfix=str(type)):
                batch_features = features[batch:batch+batch_size]
                orig_length = len(batch_features)
                batch_features = batch_features + [0] * (batch_size - len(batch_features))
                feature_effectss = jax.vmap(lambda x: circuitizer.compute_feature_effects(feature_type, layer, x, mask, layer_window=1, position=None))(jnp.asarray(batch_features))
                # feature_effectss = circuitizer.compute_feature_effects(feature_type, layer, batch_features, mask, layer_window=1)
                top_effects = defaultdict(list)
                for key, featuress in feature_effectss.items():
                    for elem, feature_effects in enumerate(featuress):
                        if elem >= orig_length:
                            continue
                        if feature_effects.ndim == 0:
                            top_effects[elem].append((float(feature_effects), key, 0))
                            continue

                        nodes_to_keep = circuit_node_dict.get(key, np.empty(0, dtype=np.int32))
                        effects = feature_effects[nodes_to_keep]
                        for idx, effect in zip(nodes_to_keep, effects):
                            top_effects[elem].append((float(effect), key, int(idx)))
                for elem, effects in top_effects.items():
                    effects.sort(reverse=True)
                    edges = effects
                    graph.extend([(weight,  key + (upstream_feature,), (type[1], type[2], type[0], batch_features[elem],) ) for weight, key, upstream_feature in edges])
                


        combined_ies = [
            (type, layer, mask, idx, weight) for layer, mask, type, idx, weight in combined_ies
        ] 


        sorted_graph = sorted(graph, reverse=True, key=lambda x: x[0])

        n_nodes = sum(map(len, important_feats_masks.values()))
        k_connections = 4
        weight_threshold = sorted_graph[n_nodes * k_connections][0]

    # %%
    if not average_over_positions:
        important_feats_masks = {}
        for mask in circuitizer.masks:
            important_feats_masks[mask] = [
                (type, layer, feat, pos) for layer, f_mask, type, feat, pos, _ in combined_ies if f_mask == mask
                ]


        flat_feats = defaultdict(list)
        for k, v in important_feats_masks.items():
            for type, layer, feat, pos in v:
                flat_feats[(k, type, layer)].append((pos, feat))


        circuit_node_dict

        graph = []

        batch_size = 16
        # k = 32
        for type, features in tqdm(sorted(flat_feats.items(), key=lambda x: (-x[0][-1], x[0][-2], x[0][-3]))):
            mask, feature_type, layer = type
            mask = jnp.array(list(circuitizer.masks.keys()).index(mask))
            for batch in trange(0, len(features), batch_size, postfix=str(type)):
                batch_features = features[batch:batch+batch_size]
                orig_length = len(batch_features)
                batch_features = batch_features + [(0, 0)] * (batch_size - len(batch_features))
                feature_effectss = jax.vmap(lambda x: circuitizer.compute_feature_effects(feature_type, layer, x[1], mask, layer_window=1, position=x[0]))(jnp.asarray(batch_features))
                # feature_effectss = circuitizer.compute_feature_effects(feature_type, layer, batch_features, mask, layer_window=1)
                top_effects = defaultdict(list)
                for key, featuress in feature_effectss.items():
                    nodes_to_keep = circuit_node_dict.get(key, np.empty((0, 2), dtype=np.int32))

                    for elem, feature_effects in enumerate(featuress):
                        if elem >= orig_length:
                            continue
                        if feature_effects.ndim == 1:
                            for idx, _ in nodes_to_keep:
                                top_effects[elem].append((float(feature_effects[idx]), key, 0, idx))
                            continue
                        effects = feature_effects[nodes_to_keep[:, 0], nodes_to_keep[:, 1]]

                        for idx, effect in zip(nodes_to_keep, effects):
                            top_effects[elem].append((float(effect), key, int(idx[1]), int(idx[0])))

                        
                for elem, effects in top_effects.items():
                    effects.sort(reverse=True)
                    edges = effects
                    graph.extend([(weight,  key + (upstream_feature,upos,), (type[1], type[2], type[0], batch_features[elem][1], batch_features[elem][0],) ) for weight, key, upstream_feature, upos in edges])
                


        combined_ies = [
            (type, layer, mask, idx, pos, weight) for layer, mask, type, idx, pos, weight in combined_ies
        ] 


        sorted_graph = sorted(graph, reverse=True, key=lambda x: x[0])

        n_nodes = sum(map(len, important_feats_masks.values()))
        k_connections = 4
        weight_threshold = sorted_graph[n_nodes * k_connections][0]

    # %%
    if average_over_positions:

        _graph = [
            (w, l, (*r[:-1], int(r[-1]))) for w, l, r in sorted_graph
        ]
    else:
        _graph = [
            (w, (*l[:-2], int(l[-2]), int(l[-1])), (*r[:-2], int(r[-2]), int(r[-1]))) for w, l, r in sorted_graph
        ]

    # %%
    if average_over_positions:
        _combined_ies = [
            (type, layer, mask, int(idx), weight) for type, layer, mask, idx, weight in combined_ies
        ]
    else:
        _combined_ies = [
            (type, layer, mask, int(idx), int(pos), float(weight)) for type, layer, mask, idx, pos, weight in combined_ies
        ]

    # %%
    tokens_decoded = [tokenizer.convert_ids_to_tokens(x) for x in circuitizer.train_tokens]
    tokens_decoded = [[x for x in y if x != "<pad>"] for y in tokens_decoded]
    tokens_decoded = [[x.replace("Ġ", " ") for x in y] for y in tokens_decoded]
    tokens_decoded = [[x.replace("▁", " ") for x in y] for y in tokens_decoded]
    tokens_decoded = [[x.replace("\n", " ") for x in y] for y in tokens_decoded]

    # %%
    if not average_over_positions:

        position_maps = defaultdict(defaultdict)

        for layer, mask, type, idx, pos, weight in _combined_ies:
            partial_id = (layer, mask, type, idx)
            partial_id = ":".join(str(x) for x in partial_id)
            position_maps[partial_id][pos] = weight

    # %%
    import json
    if average_over_positions:
        with open(f"micrlhf-progress/graph-rebirth-{task_name}_faith_0.6_l{min(layers)}_l{max(layers)}.json", 'w') as f:
            json.dump({"edges": _graph, "nodes": _combined_ies, "threshold": weight_threshold, "tokens": None}, f)
    else:
        with open(f"micrlhf-progress/graph-rebirth-{task_name}_faith_{target_faithfullness}_non_aop_n_shot_{n_shot}_l{min(layers)}_l{max(layers)}_mean_{mean_ablate}.json", 'w') as f:
            json.dump({"edges": _graph, "nodes": _combined_ies, "threshold": weight_threshold, "tokens": tokens_decoded, "position_maps": position_maps}, f)



from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str)
    args = parser.parse_args()
    main(args.task)