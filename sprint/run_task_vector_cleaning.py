
import penzai
from penzai import pz
import os
if "models" not in os.listdir("."):
    os.chdir("..")

import json

from matplotlib import pyplot as plt
from tqdm.auto import tqdm, trange
import jax.numpy as jnp
import numpy as np
import random
from penzai.data_effects.side_output import SideOutputValue
from micrlhf.utils.activation_manipulation import add_vector

from micrlhf.llama import LlamaTransformer
llama = LlamaTransformer.from_pretrained("models/gemma-2b-it.gguf", from_type="gemma", load_eager=True)

from transformers import AutoTokenizer
import jax


tokenizer = AutoTokenizer.from_pretrained("alpindale/gemma-2b")
tokenizer.padding_side = "right"


from sprint.task_vector_utils import load_tasks, ICLDataset, ICLSequence
tasks = load_tasks()



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

from micrlhf.utils.load_sae import get_nev_it_sae_suite


sep = 3978
pad = 0




task_names = [x for x in tasks]
n_seeds = 10

# n_few_shots, batch_size, max_seq_len = 64, 64, 512
n_few_shots, batch_size, max_seq_len = 20, 16, 256

prompt = "Follow the pattern:\n{}"


from sprint.task_vector_utils import ICLRunner, logprob_loss, get_tv, make_act_adder, weights_to_resid

from safetensors import safe_open
from sprint.task_vector_utils import FeatureSearch
from micrlhf.utils.ito import grad_pursuit

seed = 10

layers = list(range(1, 18))
# layers = [10, 12, 14]

# layer = 12
for task in tqdm(task_names):
    pairs = list(tasks[task].items())

    n_shot = n_few_shots-1
    if task.startswith("algo"):
        n_shot = 16

    runner = ICLRunner(task, pairs, batch_size=batch_size, n_shot=n_shot, max_seq_len=max_seq_len, seed=seed, prompt=prompt)


    tokenized = runner.get_tokens([
        x[:n_shot] for x in runner.train_pairs
    ], tokenizer)

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
        
        pr, _, rtv = sae_encode(sae, tv)

        add_act = make_act_adder(llama, rtv.astype('bfloat16'), tokens, layer, length=1, shift= 0, sep=sep)

        logits = add_act(inputs)

        recon_loss = logprob_loss(
            logits.unwrap("batch", "seq", "vocabulary"), tokens, shift=0, n_first=2, sep=sep, pad_token=0
        )

        print(
            f"Recon TV: {task}, L: {layer}, Loss: {recon_loss}"  
        )

        _, gtv = grad_pursuit(tv, sae["W_dec"], 20)

        add_act = make_act_adder(llama, gtv.astype('bfloat16'), tokens, layer, length=1, shift= 0, sep=sep)

        logits = add_act(inputs)

        ito_loss = logprob_loss(
            logits.unwrap("batch", "seq", "vocabulary"), tokens, shift=0, n_first=2, sep=sep, pad_token=0
        )

        print(
            f"Grad pursuit TV: {task}, L: {layer}, Loss: {ito_loss}"
        )

        fs = FeatureSearch(task, pairs, layer, llama, tokenizer, n_shot=1, seed=seed+100, init_w=pr, early_stopping_steps=200, n_first=2, sep=sep, pad_token=0, sae_v=8, sae=sae, batch_size=24, iterations=1000, prompt=prompt, l1_coeff=0.05)

        w, m = fs.find_weights()

        _, _, recon = sae_encode(sae, None, pre_relu=w)
        
        recon = recon.astype('bfloat16')

        add_act = make_act_adder(llama, recon, tokens, layer, length=1, shift= 0, sep=sep)

        logits = add_act(inputs)

        loss = logprob_loss(
            logits.unwrap("batch", "seq", "vocabulary"), tokens, shift=0, n_first=2, sep=sep, pad_token=0
        )

        print(
            f"Recon fs: {task}, L: {layer}, Loss: {loss}"  
        )

        with open("cleanup_results_final.jsonl", "a") as f:
            item = {
                "task": task,
                "weights": w.tolist(),
                "loss": loss.tolist(),
                "recon_loss": recon_loss.tolist(),
                "ito_loss": ito_loss.tolist(),
                "tv_loss": tv_loss.tolist(),
                "zero_loss": zero_loss.tolist(),
                "tv": tv.tolist(),
                "layer": layer
            }

            f.write(json.dumps(item) + "\n")

