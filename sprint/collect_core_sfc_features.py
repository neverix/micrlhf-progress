import json
import os
if "models" not in os.listdir("."):
    os.chdir("..")

import jax
from penzai import pz
import numpy as np


from sprint.task_vector_utils import load_tasks, ICLRunner
tasks = load_tasks()


devices = jax.devices("tpu")

def main(task_name, core):
    device = devices[core]
    print(device)
    with jax.default_device(device):
        from micrlhf.llama import LlamaTransformer
        llama = LlamaTransformer.from_pretrained("models/gemma-2b-it.gguf", from_type="gemma", load_eager=True, device_map=f"tpu:{core}")
        from sprint.icl_sfc_utils import Circuitizer

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("alpindale/gemma-2b")
        tokenizer.padding_side = "right"

    
        task = tasks[task_name]

        pairs = list(task.items())

        batch_size = 8 
        n_shot=16
        if task_name.startswith("algo"):
            n_shot = 12

        max_seq_len = 128
        seed = 10

        prompt = "Follow the pattern:\n{}"

        runner = ICLRunner(task_name, pairs, batch_size=batch_size, n_shot=n_shot, max_seq_len=max_seq_len, seed=seed, prompt=prompt)


        layers = list(range(11, 17))
        circuitizer = Circuitizer(llama, tokenizer, runner, layers, prompt)

        thresholds = np.logspace(-3, 0, 250)
        ablated_metrics, n_nodes_counts = circuitizer.run_ablated_metrics(thresholds, inverse=True)

        target_metric = (max(ablated_metrics) - min(ablated_metrics)) * 0.5 + min(ablated_metrics)

        print(target_metric)
        target_threshold = [threshold for threshold, metric in list(zip(thresholds, ablated_metrics)) if metric > target_metric][0]
        print(target_threshold)


        layers = circuitizer.layers
        selected_threshold = target_threshold


        ablation_masks = {}

        for layer in layers:
            mask_attn_out, _ = circuitizer.mask_ie(circuitizer.ie_attn[layer], selected_threshold, None, inverse=True)
            mask_resid, _ = circuitizer.mask_ie(circuitizer.ie_resid[layer], selected_threshold, None, inverse=True)
            try:
                mask_transcoder, _ = circuitizer.mask_ie(circuitizer.ie_transcoder[layer], selected_threshold, None, inverse=True)
            except KeyError:
                mask_transcoder = None

            ablation_masks[layer] = {
                "attn_out": mask_attn_out,
                "resid": mask_resid,
                "transcoder": mask_transcoder
            }

        ablated_nodes = []

        for layer, masks in ablation_masks.items():
            for mask_type, mask in masks.items():
                if mask is not None:
                    for token_type, mask in mask.items():
                            deleted = (1 - mask)
                            node_ids = np.where(deleted)[0]

                            for node_id in node_ids:
                                ablated_nodes.append((layer, mask_type, token_type, node_id.tolist()))


        typed_ies = {
            "r": circuitizer.ie_resid,
            "a": circuitizer.ie_attn,
            "t": circuitizer.ie_transcoder,
        }

        ablated_nodes_with_ie = []

        for node in ablated_nodes:
            layer, sae_type, token_type, node_id = node
            node = (int(layer), sae_type, token_type, int(node_id))
            ies = typed_ies[sae_type[0]][layer]
            masked_ies = circuitizer.mask_average(ies, token_type)
            ablated_nodes_with_ie.append(node + (masked_ies[node_id].tolist(),))


        with open(f"micrlhf-progress/top_nodes_sm_{core}.jsonl", "a") as f:
            f.write(json.dumps({
                "task_name": task_name,
                "ablated_nodes_with_ie": ablated_nodes_with_ie,
                "threshold": target_threshold
            }) + "\n")


        # return ablated_nodes_with_ie


def run_in_parallel(task_names, core):
    # Limit to the number of cores if task_names exceed
    tasks_to_run = task_names
    for task_name in tasks_to_run:
        main(task_name, core)

from threading import Thread

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("task_names", type=str)
    args = parser.parse_args()

    print(args.task_names)

    task_lists = [
        [x] for x in args.task_names.split(",")
    ]
    cores = len(task_lists)

    threads = [Thread(target=run_in_parallel, args=(task_lists[i], i)) for i in range(cores)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()





