# %%
import os
if "models" not in os.listdir("."):
    os.chdir("..")

# %%
# %load_ext autoreload
# %autoreload 2
import penzai
from penzai import pz
# pz.ts.register_as_default()
# pz.ts.register_autovisualize_magic()
# pz.enable_interactive_context()

# %%
import jax.numpy as jnp
import jax
import json

from tqdm.auto import tqdm

# %%
from micrlhf.llama import LlamaTransformer
from transformers import AutoTokenizer


filename = "models/phi-3-16.gguf"
llama = LlamaTransformer.from_pretrained(filename, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
tokenizer.padding_side = "right"

# %%
from micrlhf.utils.activation_manipulation import replace_activation

def benchmark_vector(vector, tokens, model, positions, replacement_layer):
    act_rep = replace_activation(model, vector, positions, layer=replacement_layer)

    return act_rep(tokens)

# %%
def tokens_to_inputs(tokens):
    token_array = jnp.asarray(tokens)
    token_array = jax.device_put(token_array, jax.sharding.NamedSharding(llama.mesh, jax.sharding.PartitionSpec("dp", "sp")))
    token_array = pz.nx.wrap(token_array, "batch", "seq").untag("batch").tag("batch")

    inputs = llama.inputs.from_basic_segments(token_array)
    return inputs

# %%
def logits_to_loss(logits, tokens, answer_start, pad_token=32000):
    logits = jax.nn.log_softmax(logits)

    logits = logits[:, :-1]
    logits = jnp.take_along_axis(logits, tokens[:, 1:, None], axis=-1).squeeze(-1)

    mask = tokens[:, 1:] != pad_token

    mask[:, :answer_start-1] = False

    logits = logits * mask

    return -logits.sum(axis=-1) / mask.sum(axis=-1)


# %%
from micrlhf.utils.load_sae import get_sae

MIN_SCALE = 0
MAX_SCALE = 200
def benchmark_feature(prompt_template, token_to_replace, layer, feature, explanation, batch_size=64, min_scale=MIN_SCALE, max_scale=MAX_SCALE, max_length=64, replacement_layer=2):
    vector = get_sae(layer, 4)["W_dec"][feature]
    vector = vector[None, :] * jnp.linspace(min_scale, max_scale, batch_size)[:, None]

    prompt = prompt_template.format(explanation)
    text = [prompt for _ in range(batch_size)]

    tokenized = tokenizer(text, return_tensors="np", padding="max_length", max_length=max_length, truncation=True)

    tokens = tokenized["input_ids"]

    inputs = tokens_to_inputs(tokens)

    positions = [i for i, a in enumerate(tokenizer.encode(prompt_template)) if tokenizer.decode([a]) == token_to_replace]

    logits = benchmark_vector(
        vector, inputs, llama, positions, replacement_layer
    )
    # logits_base = benchmark_vector(
    #     vector * 0.5, inputs, llama, positions, replacement_layer
    # )

    logits = logits.unwrap(
        "batch", "seq", "vocabulary"
    )

    answer_start = len(tokenizer.encode(prompt_template.partition("{}")[0]))

    loss = logits_to_loss(logits, tokens, answer_start)
    logprobs = jax.nn.log_softmax(logits)
    entropies = -jnp.sum(logprobs * jnp.exp(logprobs), axis=-1)
    # print(tokenizer.decode([tokenizer.encode(prompt)[answer_start - 1]]))
    entropy_first = entropies[:, answer_start - 1]
    max_logprob_first = jnp.max(logprobs[:, answer_start - 1], axis=-1)

    best_idx = jnp.argmin(loss)

    return loss, entropy_first, max_logprob_first, jnp.linspace(min_scale, max_scale, batch_size)[best_idx], loss[best_idx]

# %%
prompt_template = "<|user|>\nWhat is the meaning of the word \"X\"?<|end|>\n<|assistant|>\nThe meaning of the word \"X\" is \"{}\""
token_to_replace = "X"

positions = [i for i, a in enumerate(tokenizer.encode(prompt_template)) if tokenizer.decode([a]) == token_to_replace]

# %%
from datasets import load_dataset

feature_dataset = load_dataset("kisate-team/feature-explanations", split="train")

# %%
replacement_layer = 2

for i in tqdm(list(range(len(feature_dataset)))):
    item = feature_dataset[i]
    
    layer = item["layer"]
    feature = item["feature"]
    explanation = item["explanation"]

    result = {"id": i}

    if explanation is not None:
        (loss, entropy_first, max_logprob_first,
            scale, best_loss) = benchmark_feature(prompt_template, token_to_replace, layer, feature, explanation, replacement_layer=replacement_layer)
        result["loss"] = loss.tolist()
        result["scale"] = float(scale)
        result["best_loss"] = float(best_loss)
        result["entropy_first"] = entropy_first.tolist()
        result["max_logprob_first"] = max_logprob_first.tolist()

    with open("results_new.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")

# %%
with open("results_new.jsonl", "r") as f:
    results = [json.loads(l) for l in f]

# %%
results = [r for r in results if "loss" in r]
results = [r for r in results if r["scale"] > 0.0]

# %%
n_repeats = 4
batch_size = 64
prompt = "<|user|>\nWhat is the meaning of the word \"X\"?<|end|>\n<|assistant|>\nThe meaning of the word \"X\" is \""
msl = 64

# %%
from itertools import batched
from micrlhf.sampling import sample

generated_explanations = []

for batch in tqdm(list(batched(results, batch_size))):
    idx = [x["id"] for x in batch]
    dataset_sample = feature_dataset.select(idx)

    scales = [[r["scale"] for _ in range(n_repeats)] for r in batch]
    scales = [x for y in scales for x in y]
    scales = jnp.array(scales)

    vectors = []

    for r in dataset_sample:
        layer = r["layer"]
        feature = r["feature"]
        vector = get_sae(layer, 4)["W_dec"][feature]
        vectors.extend([vector for _ in range(n_repeats)])


    vectors = jnp.array(vectors)

    prompt = "<|user|>\nWhat is the meaning of the word \"X\"?<|end|>\n<|assistant|>\nThe meaning of the word \"X\" is \""

    act_rep = replace_activation(llama, vectors * scales[:, None], prompt=prompt, tokenizer=tokenizer, layer=2)

    texts = sample(act_rep, tokenizer, prompt, batch_size=batch_size*n_repeats, do_sample=True, max_seq_len=msl)[0]

    new_explanations = [
        {"id": r["id"], "explanation": [y[len(prompt):] for y in x]} for r, x in zip(batch, batched(texts, n_repeats)) 
    ]

    generated_explanations.extend(new_explanations)

    with open("gen_explanations_new.jsonl", "a") as f:
        for x in new_explanations:
            f.write(json.dumps(x) + "\n")
        # f.write(json.dumps(result) + "\n")

    # break

