import os
import json
import numpy as np
from tqdm import tqdm

if "models" not in os.listdir("."):
    os.chdir("..")

import penzai
from penzai import pz

from micrlhf.llama import LlamaTransformer
llama = LlamaTransformer.from_pretrained("models/gemma-2b-it.gguf", from_type="gemma", load_eager=True, device_map="tpu:0")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("alpindale/gemma-2b")
tokenizer.padding_side = "right"

from sprint.icl_sfc_utils import Circuitizer
from sprint.task_vector_utils import load_tasks, ICLRunner

# Load tasks
tasks = load_tasks()

batch_size = 8 
n_shot = 12
max_seq_len = 128
seed = 10

# List of task names
task_names = list(tasks.keys())

# Prepare output file for jsonl
output_filepath = "task_faithfulness_metrics.jsonl"

# Initialize tqdm for progress tracking
# task_pairs_progress = tqdm(total=len(task_names) * (len(task_names)), desc="Processing task pairs")


def main(task_name):

    # Load and prepare first task
    first_task = tasks[task_name]
    first_pairs = list(first_task.items())
    prompt = "Follow the pattern:\n{}"
    layers = list(range(11, 17))
    n_few_shot = n_shot
    if task_name.startswith("algo"):
        n_few_shot = 8
    
    # Define first runner and circuitizer
    first_runner = ICLRunner(task_name, first_pairs, batch_size=batch_size, n_shot=n_few_shot, max_seq_len=max_seq_len, seed=seed, prompt=prompt, use_same_examples=False, use_same_target=False)
    circuitizer = Circuitizer(llama, tokenizer, first_runner, layers, prompt=prompt)

    # Calculate original and zero metrics for the first task
    first_orig_metric = circuitizer.ablated_metric(llama).tolist()
    first_zero_metric = circuitizer.run_ablated_metrics([100000], layers=layers)[0][0]

    # Log thresholds and metrics settings
    thresholds = np.logspace(-5, 0, 200)
    topks = [4, 6, 12, 16, 24, 32]

    inverse = True
    do_abs = False
    mean_ablate = False
    average_over_positions = True

    # 1. Metrics for first_runner on first_task, while ablating using second_runner
    first_ablated_metrics, first_n_nodes_counts = circuitizer.run_ablated_metrics(
        thresholds, 
        inverse=inverse, 
        do_abs=do_abs, 
        mean_ablate=mean_ablate, 
        average_over_positions=average_over_positions,
        token_prefix=None, 
        layers=layers,
    )
    first_faithfullness = (np.array(first_ablated_metrics) - first_zero_metric) / (first_orig_metric - first_zero_metric)

    # Save metrics data for first runner
    first_metrics_data = {
        "task": first_task,
        "inverse": inverse,
        "orig_metric": first_orig_metric,
        "zero_metric": first_zero_metric,
        "thresholds": thresholds.tolist(),
        "n_nodes_counts": first_n_nodes_counts,
        "ablated_metrics": first_ablated_metrics,
        "faithfullness": first_faithfullness.tolist(),
        "layers": layers
    }

        # Save both results in the JSON Lines file
    with open(output_filepath, 'a') as jsonl_file:
        jsonl_file.write(json.dumps(first_metrics_data) + "\n")


        # task_pairs_progress.update(1)

    inverse = False
    do_abs = False
    mean_ablate = False
    average_over_positions = True

    # 1. Metrics for first_runner on first_task, while ablating using second_runner
    first_ablated_metrics, first_n_nodes_counts = circuitizer.run_ablated_metrics(
        thresholds, 
        inverse=inverse, 
        do_abs=do_abs, 
        mean_ablate=mean_ablate, 
        average_over_positions=average_over_positions,
        token_prefix=None, 
        layers=layers,
    )
    first_faithfullness = (np.array(first_ablated_metrics) - first_zero_metric) / (first_orig_metric - first_zero_metric)

    # Save metrics data for first runner
    first_metrics_data = {
        "task": first_task,
        "inverse": inverse,
        "orig_metric": first_orig_metric,
        "zero_metric": first_zero_metric,
        "thresholds": thresholds.tolist(),
        "n_nodes_counts": first_n_nodes_counts,
        "ablated_metrics": first_ablated_metrics,
        "faithfullness": first_faithfullness.tolist(),
        "layers": layers
    }

        # Save both results in the JSON Lines file
    with open(output_filepath, 'a') as jsonl_file:
        jsonl_file.write(json.dumps(first_metrics_data) + "\n")
        # task_pairs_progress.update(1)


from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("task_name", type=str, help="Name of the task to run the circuit ablation on.")
    args = parser.parse_args()

    main(args.task_name)
