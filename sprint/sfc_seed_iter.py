
import os
if "models" not in os.listdir("."):
    os.chdir("..")

import penzai
import jax_smi
from penzai import pz
import jax



from sprint.icl_sfc_utils import Circuitizer


from micrlhf.llama import LlamaTransformer
llama = LlamaTransformer.from_pretrained("models/gemma-2b-it.gguf", from_type="gemma", load_eager=True, device_map="tpu:0")


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("alpindale/gemma-2b")
tokenizer.padding_side = "right"


from sprint.task_vector_utils import load_tasks, ICLRunner
tasks = load_tasks()


def main(task_name, seed):
    task = tasks[task_name]

    pairs = list(task.items())

    batch_size = 8 
    n_shot=12
    max_seq_len = 64

    prompt = "Follow the pattern:\n{}"

    runner = ICLRunner(task_name, pairs, batch_size=batch_size, n_shot=n_shot, max_seq_len=max_seq_len, seed=seed, prompt=prompt, use_same_examples=False)
    
    layers = list(range(5, 18))
    circuitizer = Circuitizer(llama, tokenizer, runner, layers, prompt)

    # layers = [10,11,12,13,14,15,16]
    # layers = [8,9,10]
    layers = list(range(8, 18))
    mean_ablate = False

    orig_metric = circuitizer.ablated_metric(llama).tolist()
    zero_metric = circuitizer.run_ablated_metrics([100000], mean_ablate=mean_ablate, layers=layers)[0][0]

    print(orig_metric, zero_metric)

    
    import numpy as np
    # thresholds = np.linspace(0, 1e-4, 100)
    # thresholds = np.linspace(1.4 * 1e-4, 1.45 * 1e-4, 200)
    # thresholds = np.logspace(-4, -1, 150)
    thresholds = np.logspace(-7, 0, 200)
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

    
    # target_faithfullness = 1

    # target_threshold = [threshold for threshold, metric in reversed(list(zip(thresholds, faithfullness))) if metric > target_faithfullness][0]

    target_threshold = 0.03

    
    from tqdm.auto import tqdm

    # layers = circuitizer.layers
    # layers = [15,16]
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

    
    circuit_nodes_with_ies[:10]

    
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
                    ies_mask = circuitizer.mask_average(ies, mask, average_over_positions=average_over_positions)
                    # print(ies_mask.tolist())
                    # raise

                    if average_over_positions:
                        combined_ies.append((layer, mask, type, 0, ies_mask.tolist()))

                    else:
                        for pos, ie in enumerate(ies_mask):
                            if ie > selected_threshold:
                                combined_ies.append((layer, mask, type, 0, pos, ie))

    
    combined_ies = sorted(combined_ies, key=lambda x: -x[-1])

    
    
    import jax.numpy as jnp
    from tqdm.auto import trange

    if average_over_positions:
        combined_ies = [
            (type, layer, mask, idx, weight) for layer, mask, type, idx, weight in combined_ies
        ] 


    
    if not average_over_positions:

        combined_ies = [
            (type, layer, mask, idx, pos, weight) for layer, mask, type, idx, pos, weight in combined_ies
        ] 

    
    if average_over_positions:
        _combined_ies = [
            (type, layer, mask, int(idx), weight) for type, layer, mask, idx, weight in combined_ies
        ]
    else:
        _combined_ies = [
            (type, layer, mask, int(idx), int(pos), float(weight)) for type, layer, mask, idx, pos, weight in combined_ies
        ]
    
    with open(
        f"micrlhf-progress/combined-ies-seeds.jsonl", 'a'
    ) as f:
        import json
        entry = {
            "task_name": task_name,
            "seed": seed,
            "layers": layers,
            "average_over_positions": average_over_positions,
            "combined_ies": _combined_ies
        }

        f.write(json.dumps(entry) + "\n")


from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--task_name", type=str)
parser.add_argument("--seed", type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.task_name, args.seed)