#!/usr/bin/env python
# coding: utf-8

# !python tvc_tune.py && python tvc_tune.py phi && python tvc_tune.py g2 && python tvc_tune.py g2 big

# In[ ]:


from penzai import pz
import os
if "models" not in os.listdir("."):
    os.chdir("..")



# In[ ]:


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

# In[ ]:


from sprint.task_vector_utils import load_tasks, ICLDataset, ICLSequence
tasks = load_tasks()

task_names = list(tasks.keys())


devices = jax.devices("tpu")

# layer = 12
def main(task_names, use_phi, use_g2, use_g2b, big_g2, core):
    device = devices[core]
    print(device)
    with jax.default_device(device):
        from micrlhf.llama import LlamaTransformer
        if use_phi:
            print("Using Phi")
            llama = LlamaTransformer.from_pretrained("models/phi-3-16.gguf", load_eager=True, device_map=f"tpu:{core}")
        elif use_g2:
            print("Using Gemma 2")
            llama = LlamaTransformer.from_pretrained("models/gemma-2-2b-it.gguf", from_type="gemma2", load_eager=True, device_map=f"tpu:{core}")
        elif use_g2b:
            print("Using Gemma 2 Big")
            llama = LlamaTransformer.from_pretrained("models/gemma-2-9b-it.gguf", from_type="gemma2", load_eager=True, device_map="auto:mp=4")
        else:
            print("Using Gemma")
            llama = LlamaTransformer.from_pretrained("models/gemma-2b-it.gguf", from_type="gemma", load_eager=True, device_map=f"tpu:{core}")

        from transformers import AutoTokenizer


        if use_phi:
            tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        elif use_g2 or use_g2b:
            tokenizer = AutoTokenizer.from_pretrained("alpindale/gemma-2b")
        else:
            tokenizer = AutoTokenizer.from_pretrained("alpindale/gemma-2b")
        tokenizer.padding_side = "right"




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



        from sprint.task_vector_utils import ICLRunner, logprob_loss, get_tv, make_act_adder

        from micrlhf.utils.load_sae import sae_encode

        from safetensors import safe_open

        from micrlhf.utils.load_sae import get_nev_it_sae_suite, get_sae, get_dm_res_sae


        if use_phi:
            sep = 1599
        elif use_g2 or use_g2b:
            sep = 3978
        else:
            sep = 3978
        pad = 0

        # task_names = ["antonyms"]
        n_seeds = 10

        # n_few_shots, batch_size, max_seq_len = 64, 64, 512
        n_few_shots, batch_size, max_seq_len = 32, 16, 256 if not (use_phi or use_g2 or use_g2b) else 256

        prompt = "Follow the pattern:\n{}"
        if use_phi:
            prompt = "<|user|>\n" + prompt
        elif use_g2b:
            prompt = "<start_of_turn>user\n" + prompt


        from sprint.task_vector_utils import ICLRunner, logprob_loss, get_tv, make_act_adder, weights_to_resid

        from safetensors import safe_open
        from sprint.task_vector_utils import FeatureSearch
        from micrlhf.utils.ito import grad_pursuit


        from collections import defaultdict

        seed = 10

        # layers = list(range(1, 18))
        # layers = [10, 12, 14]

        if use_phi:
            layers = [16]
        elif use_g2:
            layers = [16]
        elif use_g2b:
            layers = [20]
        else:
            layers = [12]

        sweep_results = defaultdict(lambda: defaultdict(dict))
        l1_coeffs = [1e-5, 1e-4, 1e-3, 1e-2, 2.5e-2, 5e-2, 1e-1]
        k_tvs = [5, 10, 20, 30, 40, 50, 100]

        save_to = f"data/l1_sweep_results_{'phi' if use_phi else 'gemma2' if use_g2 else 'gemma2-9b' if use_g2b else 'gemma'}{'_big' if big_g2 else ''}_{core}.json"
        if os.path.exists(save_to):
            sweep_results = json.load(open(save_to))
            sweep_results = {k: defaultdict(dict, v) for k, v in sweep_results.items()}
            sweep_results = defaultdict(lambda: defaultdict(dict), sweep_results)

        for task in tqdm(sorted(task_names)):
            pairs = list(tasks[task].items())
            n_shot = n_few_shots-1
            if task.startswith("algo"):
                n_shot = 16

            runner = ICLRunner(task, pairs, batch_size=batch_size*1, n_shot=n_shot, max_seq_len=max_seq_len,
                            seed=seed, prompt=prompt)


            try:
                tokenized = runner.get_tokens([
                    x[:n_shot] for x in runner.train_pairs
                ], tokenizer)
            except AssertionError:
                print("Warning: Skipping", task)
                return

            inputs = tokenized_to_inputs(**tokenized)
            train_tokens = tokenized["input_ids"]

            _, all_resids = get_resids_call(inputs)

            tokenized = runner.get_tokens(runner.eval_pairs, tokenizer)
            inputs = tokenized_to_inputs(**tokenized)
            tokens = tokenized["input_ids"]

            logits = llama(inputs)
            
            zero_loss = logprob_loss(
                logits.unwrap("batch", "seq", "vocabulary"), tokens, shift= 0, n_first=2, sep=sep, pad_token=0
            )

            print(
                f"Zero: {task}, Loss: {zero_loss}"  
            )

            for layer in layers:
                key = f"{task}:{layer}"
                sweep_results[key]["Zero"][0] = (0, float(zero_loss))
                json.dump(sweep_results, open(save_to, "w"))
                continue
            
                if use_phi:
                    sae = get_sae(layer)
                elif use_g2:
                    sae = get_dm_res_sae(layer, load_65k=big_g2)
                elif use_g2b:
                    sae = get_dm_res_sae(layer, load_65k=big_g2, nine_b=True)
                else:
                    sae = get_nev_it_sae_suite(layer)

                resids = all_resids[layer].value.unwrap(
                    "batch", "seq", "embedding"
                )

                tv = get_tv(resids, train_tokens, shift = 0, sep=sep)

                add_act = make_act_adder(llama, tv.astype('bfloat16'), tokens, layer, length=1, shift= 0, sep=sep)

                logits = add_act(inputs)

                tv_loss = logprob_loss(
                    logits.unwrap("batch", "seq", "vocabulary"), tokens, shift=0, n_first=2, sep=sep, pad_token=0
                )

                print(
                    f"TV: {task}, L: {layer}, Loss: {tv_loss}"  
                )
                
                pr, post_r, rtv = sae_encode(sae, tv)

                add_act = make_act_adder(llama, rtv.astype('bfloat16'), tokens, layer, length=1, shift= 0, sep=sep)

                logits = add_act(inputs)

                recon_loss = logprob_loss(
                    logits.unwrap("batch", "seq", "vocabulary"), tokens, shift=0, n_first=2, sep=sep, pad_token=0
                )
                
                sweep_results[key]["TV"][0] = (rtv.size, float(tv_loss))
                sweep_results[key]["SAE reconstruction"][0] = (int((post_r != 0).sum()), float(recon_loss))

                for k_tv in k_tvs:
                    _, gtv = grad_pursuit(tv, sae["W_dec"], k_tv)

                    add_act = make_act_adder(llama, gtv.astype('bfloat16'), tokens, layer, length=1, shift= 0, sep=sep)

                    logits = add_act(inputs)

                    ito_loss = logprob_loss(
                        logits.unwrap("batch", "seq", "vocabulary"), tokens, shift=0, n_first=2, sep=sep, pad_token=0
                    )
                    
                    sweep_results[key]["ITO"][k_tv] = (k_tv, float(ito_loss))
                    print("ITO:", k_tv, ito_loss)

                for l1_coeff in l1_coeffs:
                    print("L1 coefficient:", l1_coeff)
                    fs = FeatureSearch(task, pairs, layer, llama, tokenizer, n_shot=1,
                                    seed=seed+100, init_w=pr, early_stopping_steps=50,
                                    n_first=2, sep=sep, pad_token=0, sae_v=8, sae=sae,
                                    batch_size=12, iterations=1000, prompt=prompt,
                                    l1_coeff=jnp.array(l1_coeff), n_batches=1, lr=0.05)

                    w, metrics = fs.find_weights()
                    l0, t_loss, steps = int(metrics["l0"]), float(metrics["loss"]), int(metrics["step"])

                    _, _, recon = sae_encode(sae, None, pre_relu=w)
                
                    recon = recon.astype('bfloat16')

                    add_act = make_act_adder(llama, recon, tokens, layer, length=1, shift= 0, sep=sep)

                    logits = add_act(inputs)

                    loss = float(logprob_loss(
                        logits.unwrap("batch", "seq", "vocabulary"), tokens, shift=0, n_first=2, sep=sep, pad_token=0
                    ))

                    print("L0:", l0, "Loss:", loss, "Steps:", steps, "Train loss", t_loss)
                    sweep_results[key]["Task vector cleaning"][l1_coeff] = (l0, loss)
                print(sweep_results)
                json.dump(sweep_results, open(save_to, "w"))


from threading import Thread

def run_in_parallel(task_names, use_phi, use_g2, use_g2b, big_g2, core):
    # Limit to the number of cores if task_names exceed
    tasks_to_run = task_names
    main(
        tasks_to_run, use_phi, use_g2, use_g2b, big_g2, core
    )


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # parser.add_argument("task_names", type=str)
    parser.add_argument("cores", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("--big", action="store_true")
    args = parser.parse_args()

    use_phi = args.model == "phi"
    use_g2 = args.model == "g2"
    use_g2b = args.model == "g2b"
    big_g2 = args.big

    cores = int(args.cores)

    task_lists = [
        task_names[i::cores] for i in range(cores)
    ]

    cores = len(task_lists)

    threads = [Thread(target=run_in_parallel, args=(task_lists[i], use_phi, use_g2, use_g2b, big_g2, i)) for i in range(cores)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
