import json
import penzai
from penzai import pz
from matplotlib import pyplot as plt
from tqdm.auto import tqdm, trange
import jax.numpy as jnp
import numpy as np
import random
from penzai.data_effects.side_output import SideOutputValue
from micrlhf.utils.activation_manipulation import add_vector

from micrlhf.utils.vector_storage import save_and_upload_vector
from task_vector_utils import FeatureSearch

filename = "models/phi-3-16.gguf"
from micrlhf.llama import LlamaTransformer
llama = LlamaTransformer.from_pretrained(filename, device_map="auto")
from micrlhf.sampling import sample
from transformers import AutoTokenizer
import jax
# tokenizer = load_tokenizer(filename)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

from task_vector_utils import load_tasks, ICLDataset, ICLSequence
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

def generate_task_prompt(task, n_shots):
    prompt = "<user>Follow the pattern\n{}"
    examples = []

    for s, t in random.sample(list(tasks[task].items()), n_shots):
        examples.append(f"{s} -> {t}")
    prompt = prompt.format("\n".join(examples))

    # print(prompt)

    return prompt

def tokenized_to_inputs(input_ids, attention_mask):
    token_array = jnp.asarray(input_ids)
    token_array = jax.device_put(token_array, jax.sharding.NamedSharding(llama.mesh, jax.sharding.PartitionSpec("dp", "sp")))
    token_array = pz.nx.wrap(token_array, "batch", "seq").untag("batch").tag("batch")

    mask_array = jnp.asarray(attention_mask, dtype=jnp.bool)
    mask_array = jax.device_put(mask_array, jax.sharding.NamedSharding(llama.mesh, jax.sharding.PartitionSpec("dp", "sp")))
    mask_array = pz.nx.wrap(mask_array, "batch", "seq").untag("batch").tag("batch")

    inputs = llama.inputs.from_basic_segments(token_array)
    return inputs

seed = 10
n_few_shots, batch_size, max_seq_len = 40, 32, 512


from task_vector_utils import ICLRunner, logprob_loss, get_tv, make_act_adder
from micrlhf.utils.load_sae import get_sae, sae_encode_gated


task_names = [
    "en_es", "en_fr", "en_de", "en_it", "en_ru", "person_profession", "country_capital", "location_religion", "location_continent", "location_language", "es_en", "fr_en", "location_country"
]

layers = [
    14, 17
]

saes = {
    layer: get_sae(layer, 4) for layer in layers
}

results = {}

for task in tqdm(task_names):
    results[task] = {}

    n_few_shots = 40
    pairs = list(tasks[task].items())
    runner = ICLRunner(task, pairs, batch_size=32, n_shot=n_few_shots-1, max_seq_len=max_seq_len, seed=10)
    tokenized = runner.get_tokens(runner.train_pairs, tokenizer)
    inputs = tokenized_to_inputs(**tokenized)
    train_tokens = tokenized["input_ids"]

    logits, resids = get_resids_call(inputs)

    loss = logprob_loss(
        logits.unwrap("batch", "seq", "vocabulary"), train_tokens, shift=1 if task.startswith("algo") else 0, n_first=2
    )

    results[task] = {
        "full_loss": float(loss),
    }

    print(
        f"Full: {task}, loss: {loss}, n_shot: {n_few_shots}"
    )

    mask = train_tokens == 1599

    tokenized = runner.get_tokens(runner.eval_pairs, tokenizer)
    inputs = tokenized_to_inputs(**tokenized)
    tokens = tokenized["input_ids"]


    for layer in layers:
        try:
            sae = saes[layer]

            results[task][layer] = {}
            _resids = resids[layer].value.unwrap(
                "batch", "seq", "embedding"
            )

            _resids = _resids[mask]
            tv = _resids.mean(axis=0)

            add_act = make_act_adder(llama, tv.astype('bfloat16'), tokens, layer, length=1, shift= 0)

            logits = add_act(inputs)

            loss = logprob_loss(
                logits.unwrap("batch", "seq", "vocabulary"), tokens, shift=1 if task.startswith("algo") else 0, n_first=2
            )

            print(
                f"TV: {task}, L: {layer}, Loss: {loss}"  
            )

            results[task][layer]["tv_loss"] = float(loss)

            _, pr, _ = sae_encode_gated(sae, tv)

            fs = FeatureSearch(task, pairs, layer, llama, tokenizer, n_shot=1, seed=seed+100, init_w=pr, early_stopping_steps=100, n_first=2)

            w, m = fs.find_weights()

            weights = (w > 0) * jax.nn.relu(w * jax.nn.softplus(sae["s_gate"]) * sae["scaling_factor"] + sae["b_gate"])   

            recon = jnp.einsum("fv,f->v", sae["W_dec"], weights) + sae["b_dec"]
            recon = recon.astype('bfloat16')

            add_act = make_act_adder(llama, recon, tokens, layer, length=1, shift= 0)

            logits = add_act(inputs)

            loss = logprob_loss(
                logits.unwrap("batch", "seq", "vocabulary"), tokens, shift=1 if task.startswith("algo") else 0, n_first=2
            )

            print(
                f"Recon fs: {task}, L: {layer}, Loss: {loss}"  
            )

            results[task][layer]["fs_loss"] = float(loss)

            save_and_upload_vector(f"fs_{task}_{layer}_v4", w, overwrite=True)

            with open("results_wlito_3.json", "w") as f:
                json.dump(results, f)
        except Exception as e:
            print(e)
            continue

task_names = [
    "en_es", "en_fr", "en_de", "en_it", "en_ru", "person_profession", "country_capital", "location_religion", "location_continent", "location_language", "es_en", "fr_en", "location_country"
]

layers = [
    18, 17, 16, 20
]

saes = {
    layer: get_sae(layer, 9) for layer in layers
}

results = {}

for task in tqdm(task_names):
    results[task] = {}

    n_few_shots = 40
    pairs = list(tasks[task].items())
    runner = ICLRunner(task, pairs, batch_size=32, n_shot=n_few_shots-1, max_seq_len=max_seq_len, seed=10)
    tokenized = runner.get_tokens(runner.train_pairs, tokenizer)
    inputs = tokenized_to_inputs(**tokenized)
    train_tokens = tokenized["input_ids"]

    logits, resids = get_resids_call(inputs)

    loss = logprob_loss(
        logits.unwrap("batch", "seq", "vocabulary"), train_tokens, shift=1 if task.startswith("algo") else 0, n_first=2
    )

    results[task] = {
        "full_loss": float(loss),
    }

    print(
        f"Full: {task}, loss: {loss}, n_shot: {n_few_shots}"
    )

    mask = train_tokens == 1599

    tokenized = runner.get_tokens(runner.eval_pairs, tokenizer)
    inputs = tokenized_to_inputs(**tokenized)
    tokens = tokenized["input_ids"]


    for layer in layers:
        try:
            sae = saes[layer]

            results[task][layer] = {}
            _resids = resids[layer].value.unwrap(
                "batch", "seq", "embedding"
            )

            _resids = _resids[mask]
            tv = _resids.mean(axis=0)

            add_act = make_act_adder(llama, tv.astype('bfloat16'), tokens, layer, length=1, shift= 0)

            logits = add_act(inputs)

            loss = logprob_loss(
                logits.unwrap("batch", "seq", "vocabulary"), tokens, shift=1 if task.startswith("algo") else 0, n_first=2
            )

            print(
                f"TV: {task}, L: {layer}, Loss: {loss}"  
            )

            results[task][layer]["tv_loss"] = float(loss)

            _, pr, _ = sae_encode_gated(sae, tv)

            fs = FeatureSearch(task, pairs, layer, llama, tokenizer, n_shot=1, seed=seed+100, init_w=pr, early_stopping_steps=100, n_first=2, sae_v=9)

            w, m = fs.find_weights()

            weights = (w > 0) * jax.nn.relu(w * jax.nn.softplus(sae["s_gate"]) * sae["scaling_factor"] + sae["b_gate"])   

            recon = jnp.einsum("fv,f->v", sae["W_dec"], weights) + sae["b_dec"]
            recon = recon.astype('bfloat16')

            add_act = make_act_adder(llama, recon, tokens, layer, length=1, shift= 0)

            logits = add_act(inputs)

            loss = logprob_loss(
                logits.unwrap("batch", "seq", "vocabulary"), tokens, shift=1 if task.startswith("algo") else 0, n_first=2
            )

            print(
                f"Recon fs: {task}, L: {layer}, Loss: {loss}"  
            )

            results[task][layer]["fs_loss"] = float(loss)

            save_and_upload_vector(f"fs_{task}_{layer}_v9_2", w, overwrite=True)

            with open("results_wlito_v9.json", "w") as f:
                json.dump(results, f)
        except Exception as e:
            print(e)
            continue