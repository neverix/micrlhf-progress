import os
if "models" not in os.listdir("."):
    os.chdir("..")

import penzai
import jax_smi
import json
from penzai import pz

from micrlhf.llama import LlamaTransformer
llama = LlamaTransformer.from_pretrained("models/gemma-2b-it.gguf", from_type="gemma", load_eager=True, device_map="tpu:0")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("alpindale/gemma-2b")
tokenizer.padding_side = "right"

# %%
from sprint.icl_sfc_utils import Circuitizer

# %%
from sprint.task_vector_utils import load_tasks, ICLRunner
tasks = load_tasks()

def main(task_name):
    task = tasks[task_name]
    pairs = list(task.items())

    batch_size = 1 
    n_shot=12
    if task_name.startswith("algo"):
        n_shot = 8
    max_seq_len = 128
    seed = 10

    prompt = "Follow the pattern:\n{}"

    runner = ICLRunner(task_name, pairs, batch_size=batch_size, n_shot=n_shot, max_seq_len=max_seq_len, seed=seed, prompt=prompt, use_same_examples=False, use_same_target=False)


    layers = list(range(2, 18))
    circuitizer = Circuitizer(llama, tokenizer, runner, layers, prompt)

    # %%

    average_over_positions = True
    orig_metric = circuitizer.ablated_metric(llama).tolist()

    # %%
    import numpy as np

    inverse = False
    do_abs = False
    mean_ablate = False


    layers_to_check = list(range(2, 18))


    # %%
    from tqdm.auto import tqdm


    from plotly.subplots import make_subplots
    from tqdm.auto import tqdm
    import plotly.graph_objects as go
    import jax.numpy as jnp

    ies = {
        "resid": circuitizer.ie_resid,
        "attn_out": circuitizer.ie_attn,
        "transcoder": circuitizer.ie_transcoder,
    }


    mask_names = circuitizer.masks.keys()


    combined_effects = {}

    _feature_types = ["resid", "attn_out", "transcoder"]

    for layer in tqdm(layers_to_check):
        feature_types = _feature_types
        if layer == 17:
            feature_types = ["resid", "attn_out"]
        feature_masks = []
        all_node_ies = []
        all_idx = []
        for feature_type in feature_types:
            for token_type in ["input", "arrow", "output"]:
                node_ies = circuitizer.mask_average(ies[feature_type][layer], token_type)
                non_zero_idx = np.where(node_ies != 0)[0]

                other_masks = {}

                for ft in feature_types:
                    if ft != feature_type:
                        masks = {
                            k: jnp.ones_like(node_ies).astype(jnp.bool) for k in mask_names
                        }
                        other_masks[ft] = masks

                for idx in non_zero_idx:
                    masks = {
                        k: jnp.ones_like(node_ies).astype(jnp.bool) for k in mask_names
                    }
                    masks[token_type] = masks[token_type].at[idx].set(False)

                    masks = {
                        feature_type: masks,
                        **other_masks
                    }

                    feature_masks.append(masks)
                    all_node_ies.append((node_ies[idx].tolist(), token_type, feature_type))
                    all_idx.append((idx, token_type, feature_type))

        feat_metrics = circuitizer.run_ablate_feature_masks(feature_masks, layer)

        combined_effects[layer] = {
            "node_ies": all_node_ies,
            "idx": all_idx,
            "metrics": feat_metrics
        }


    from collections import defaultdict

    transformed_effects = {}

    for layer, effects in combined_effects.items():
        node_ies = effects["node_ies"]
        idx = effects["idx"]
        metrics = effects["metrics"]

        transformed_effects[layer] = defaultdict(lambda: defaultdict(lambda: {"ies": [], "metrics": [], "idx": []}))

        for ie, i, m in zip(node_ies, idx, metrics):
            token_type, feature_type = i[1:]

            transformed_effects[layer][token_type][feature_type][
                "ies"
            ].append(ie[0])

            transformed_effects[layer][token_type][feature_type][
                "metrics"
            ].append(orig_metric - m)

            transformed_effects[layer][token_type][feature_type][
                "idx"
            ].append(int(i[0]))

    with open("results_correlation.jsonl", "a") as f:
        f.write(json.dumps({
            "task_name": task_name,
            "combined_effects": transformed_effects,
        }) + "\n")



from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--task", type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.task)