from penzai import pz
import os
if "models" not in os.listdir("."):
    os.chdir("..")


import json
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import jax
import json
from tqdm.auto import tqdm, trange
import jax.numpy as jnp
import sys

from jax_smi import initialise_tracking
initialise_tracking()

from sprint.task_vector_utils import load_tasks, ICLDataset, ICLSequence
tasks = load_tasks()

task_names = list(tasks.keys())


devices = jax.devices("tpu")

layer = 12
mask_name = "arrow"

import json

with open("cleanup_results_final.jsonl") as f:
    lines = f.readlines()
    results = [json.loads(line) for line in lines]

with open("cleanup_results_algo.jsonl") as f:
    lines = f.readlines()
    results_algo = [json.loads(line) for line in lines]

import numpy as np

from micrlhf.utils.load_sae import get_nev_it_sae_suite, sae_encode, sae_encode_gated

sae = get_nev_it_sae_suite(layer=layer)

features = []
task_features = {}

for task_name in task_names:
    if not task_name.startswith("algo"):
        task_results = [result for result in results if result["task"] == task_name and result["layer"] == layer]
    else:
        task_results = [result for result in results_algo if result["task"] == task_name and result["layer"] == layer]

    for result in task_results:
        weights = np.array(result["weights"])
        # s = jax.nn.softplus(sae["s_gate"]) * sae["scaling_factor"]
        # threshold = jnp.maximum(0, sae["b_gate"] - sae["b_enc"] * s)
        # w = weights
        # w = w * (w > 0)

        _, w, _ = sae_encode(sae, None, pre_relu=weights)

        # print(threshold)

        new_features = np.nonzero(w)[0].tolist()
        features += new_features
        print(task_name, "TVC:", result["loss"], "TV:", result["tv_loss"], new_features)

        task_features[task_name] = new_features
    

features = list(set(features))
len(features)



def main(task_names, core):
    device = devices[core]
    print(device)
    with jax.default_device(device):

        from micrlhf.llama import LlamaTransformer
        llama = LlamaTransformer.from_pretrained("models/gemma-2b-it.gguf", from_type="gemma", load_eager=True, device_map=f"tpu:{core}")

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("alpindale/gemma-2b")
        tokenizer.padding_side = "right"

        from sprint.task_vector_utils import ICLRunner, logprob_loss, get_tv, make_act_adder
        from micrlhf.llama import LlamaBlock
        from micrlhf.sampling import sample, jit_wrapper


        get_resids = llama.select().at_instances_of(LlamaBlock).apply_with_selected_index(lambda i, x:
            pz.nn.Sequential([
                pz.de.TellIntermediate.from_config(tag=f"resid_pre_{i}"),
                x
            ])
        )
        get_resids = pz.de.CollectingSideOutputs.handling(get_resids, tag_predicate=lambda x: x.startswith("resid_pre"))
        get_resids_call = jit_wrapper.Jitted(get_resids)



        def tokenized_to_inputs(input_ids, attention_mask):
            token_array = jnp.asarray(input_ids)
            token_array = jax.device_put(token_array, jax.sharding.NamedSharding(llama.mesh, jax.sharding.PartitionSpec("dp", "sp")))
            token_array = pz.nx.wrap(token_array, "batch", "seq").untag("batch").tag("batch")

            mask_array = jnp.asarray(attention_mask, dtype=jnp.bool)
            mask_array = jax.device_put(mask_array, jax.sharding.NamedSharding(llama.mesh, jax.sharding.PartitionSpec("dp", "sp")))
            mask_array = pz.nx.wrap(mask_array, "batch", "seq").untag("batch").tag("batch")

            inputs = llama.inputs.from_basic_segments(token_array)
            return inputs

        import dataclasses
        from tqdm.auto import tqdm

        n_few_shots, batch_size, max_seq_len = 1, 16, 64
        seed = 10

        prompt = "Follow the pattern:\n{}"

        def calc_acc(tokens, sep, logits, runner):
            preds = logits.argmax(-1)
            preds = preds[:, :-1]
            targets = tokens[:, 1:]

            mask = tokens[:, :-1] == sep

            hits = targets == preds

            hits = hits * mask

            hits = hits.sum()
            return hits / mask.sum()


        def make_taker(llama, layer):
            taker = jit_wrapper.Jitted(llama.select().at_instances_of(LlamaBlock).apply_with_selected_index(
                lambda i, x: x if i >= layer else pz.nn.Identity()
            ).select().at_instances_of(pz.nn.EmbeddingLookup).apply(lambda _: pz.nn.Identity())
                            .select().at_instances_of(pz.nn.ConstantRescale).pick_nth_selected(0).apply(lambda _: pz.nn.Identity()))

            return taker

        taker = make_taker(llama, layer)

        from collections import defaultdict

        for task_name in tqdm(task_names):

            sep = 3978
            pad = 0


            steering_results = defaultdict(dict)

            pairs = list(tasks[task_name].items())

            n_shot = n_few_shots - 1
            if task_name.startswith("algo"):
                n_shot = 24

            runner = ICLRunner(task_name, pairs, batch_size=batch_size, n_shot=n_shot, max_seq_len=max_seq_len, seed=seed, prompt=prompt)

            tokenized = runner.get_tokens([
                x[:n_few_shots] for x in runner.train_pairs
            ], tokenizer)

            inputs = tokenized_to_inputs(**tokenized)
            train_tokens = tokenized["input_ids"]

            _, all_resids = get_resids_call(inputs)

            resids = all_resids[layer].value.unwrap("batch", "seq", "embedding")

            mask = train_tokens == sep
            col_indices = jnp.arange(mask.shape[1])
            col_indices_broadcasted = mask * col_indices
            sorted_indices = jnp.sort(col_indices_broadcasted, axis=1, descending=True)

            k = jnp.sum(mask[0]).astype(int)

            positions = sorted_indices[:, :k]
            
            def steer_with_direction(direction, scale):
                direction = direction / jnp.linalg.norm(direction)
                direction = direction * scale
                
                modified = jax.vmap(lambda a, b: a.at[b].add(direction))(
                    resids, positions
                )
                modified = pz.nx.wrap(modified, "batch", "seq", "embedding")

                _inputs = dataclasses.replace(inputs, tokens=modified)

                # _taker = add_vector(
                #     taker, direction, layer + 1, scale=1, position=positions
                # )

                logits = taker(_inputs).unwrap("batch", "seq", "vocabulary")

                acc = calc_acc(train_tokens, sep, logits, runner)

                return logprob_loss(logits, train_tokens, sep=sep, pad_token=pad, n_first=2).tolist(), acc.tolist()

            # _features = task_features[task_name]

            logits = llama(inputs)

            logits = logits.unwrap("batch", "seq", "vocabulary")

            acc = calc_acc(train_tokens, sep, logits, runner)


            for feature in tqdm(features):
                steering_results[task_name][feature] = [[steer_with_direction(-sae["W_dec"][feature], scale) for scale in np.logspace(0, 1.5, 50)]]
                steering_results[task_name][feature].append((logprob_loss(logits, train_tokens, sep=sep, pad_token=pad, n_first=2).tolist(), acc.tolist()))

            with open(f"negative_steering_results_{core}.jsonl", "a") as f:
                f.write(json.dumps(steering_results) + "\n")



from threading import Thread

def run_in_parallel(task_names, core):
    # Limit to the number of cores if task_names exceed
    tasks_to_run = task_names
    main(tasks_to_run, core)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # parser.add_argument("task_names", type=str)
    parser.add_argument("cores", type=str)
    args = parser.parse_args()

    cores = int(args.cores)

    task_lists = [
        task_names[i::cores] for i in range(cores)
    ]

    cores = len(task_lists)

    threads = [Thread(target=run_in_parallel, args=(task_lists[i], i)) for i in range(cores)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
