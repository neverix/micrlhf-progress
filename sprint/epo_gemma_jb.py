#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
if "models" not in os.listdir("."):
    os.chdir("..")


# In[2]:


# %load_ext autoreload
# %autoreload 2
import penzai
from penzai import pz
# pz.ts.register_as_default()
# pz.ts.register_autovisualize_magic()
# pz.enable_interactive_context()


# In[3]:


from micrlhf.llama import LlamaTransformer
from transformers import AutoTokenizer


filename = "models/gemma-2b-it.gguf"
llama = LlamaTransformer.from_pretrained(filename, device_map="auto", from_type="gemma", load_eager=True)
tokenizer = AutoTokenizer.from_pretrained("NousResearch/gemma-2b-it-tokenizer")
tokenizer.padding_side = "right"


# In[4]:


import numpy as np
from matplotlib import pyplot as plt
from micrlhf.sampling import sample, LlamaKVCachingTransformer
from micrlhf.utils.activation_manipulation import replace_activation, add_vector
import jax.numpy as jnp


# In[5]:


from micrlhf.llama import LlamaBlock
from micrlhf.flash import flashify
from micrlhf.sampling import sample, trange, jnp, load_tokenizer, jit_wrapper
get_resids = llama.select().at_instances_of(LlamaBlock).apply_with_selected_index(lambda i, x:
    pz.nn.Sequential([
        pz.de.TellIntermediate.from_config(tag=f"resid_pre.{i}"),
        x
    ])
)
get_resids = pz.de.CollectingSideOutputs.handling(get_resids, tag_predicate=lambda x: x.startswith("resid_pre"))
get_resids_call = jit_wrapper.Jitted(get_resids)
def rep_w_linear(mod):
    val = mod.table.embeddings.value  # vocabulary, embedding
    return pz.nn.Linear(pz.nn.Parameter(val, "input_embed"), ["vocabulary"], ["embedding"])
get_resids_one_hot = get_resids.select().at_instances_of(pz.nn.EmbeddingLookup).apply(rep_w_linear)
get_resids_one_hot_call = jit_wrapper.Jitted(get_resids_one_hot)


# In[10]:


from micrlhf.sampling import sample, trange, jnp, load_tokenizer, jit_wrapper
from tqdm.auto import trange
from penzai.toolshed import sharding_util
import dataclasses
from functools import partial
import numpy as np
import jax

@jax.jit
def loss_fn(logits, inputs):
    losses = pz.nx.nmap(lambda l, i: jnp.take_along_axis(jax.nn.log_softmax(l[:-1], -1), i[1:, None], 1)[:, 0].mean()
                        )(logits.untag("seq", "vocabulary"), inputs.tokens.untag("seq"))
    return -losses

bs_start = llama.mesh.shape["dp"]
# n_x = 16
# n_x = 19
n_x = 20
# n_x = 32
has_end = False
tokens_init = tokenizer.encode(f"<start_of_turn>user\nX{' X' * (n_x - 1)}")
optim_mask = ["X" in tokenizer.decode([token]) for token in tokens_init]
assert sum(optim_mask) == n_x
tokens_init = np.asarray(tokens_init)
MAX_ELITES = 16
PROB_SWAP = 0.1  # probability of a swap
PROB_GRADS = 0.8    # probability of using gradients 
tokens_init = np.repeat(tokens_init[None, :], MAX_ELITES, axis=0)
def tokens_to_array(tokens):
    token_array = jnp.asarray(tokens)
    if len(token_array) >= bs_start:
        token_array = jax.device_put(token_array, jax.sharding.NamedSharding(llama.mesh, jax.sharding.PartitionSpec("dp", "sp")))
    token_array = pz.nx.wrap(token_array, "batch", "seq")
    return token_array
def run_tokens(token_array, grad_metric=None):
    if not isinstance(token_array, pz.nx.NamedArray):
        token_array = tokens_to_array(token_array)
    inputs = llama.inputs.from_basic_segments(token_array)
    if grad_metric:
        @partial(jax.grad, has_aux=True)
        def lwg(x):
            logits, resids = get_resids_one_hot_call(dataclasses.replace(inputs, tokens=x))
            resids = {resid.tag: resid.value for resid in resids}
            metric = grad_metric(logits, resids, inputs)
            return metric, (logits, resids)
        vocab = llama.select().at_instances_of(pz.nn.EmbeddingLookup).get_sequence()[0].table.embeddings.value.named_shape["vocabulary"]
        one_hots = pz.nx.nmap(lambda x: jax.nn.one_hot(x, vocab))(inputs.tokens).tag("vocabulary")
        grad, (logits, resids) = lwg(one_hots)
    else:
        logits, resids = get_resids_call(inputs)
    losses = loss_fn(logits, inputs)
    if not grad_metric:
        resids = {resid.tag: resid.value for resid in resids}
    return_vals = logits, losses, resids
    if grad_metric:
        return_vals = return_vals + (grad,)
    return return_vals

mask = jax.device_put(jnp.asarray(optim_mask), jax.sharding.NamedSharding(llama.mesh, jax.sharding.PartitionSpec("sp")))

# @partial(jax.jit, static_argnames=("max_inv_temp", "expected_changes"))
@jax.jit
def temper(logits, key, elites, grads, max_inv_temp, expected_changes):
    key_choice, key_random = jax.random.split(key)
    index = jax.random.randint(key_choice, (), 0, len(logits) - 1)
    key_categorical, key_uniform, key_bernoulli, key_randint, key_use_grads, key_mutations = jax.random.split(key_random, 6)
    logit = logits[index]
    elite = elites[index]
    grads = grads[index]
    
    logit = jnp.roll(logit, 1, 0)
    logit = logit * jax.random.uniform(key_uniform, minval=0, maxval=max_inv_temp)
    use_grads = jax.random.bernoulli(key_use_grads, p=PROB_GRADS).astype(jnp.int_)
    logit = jax.lax.switch(use_grads, ((lambda x: x), (lambda x: jnp.where(grads, x, -jnp.inf))), logit)
    to_change = jax.random.bernoulli(key_bernoulli, jnp.maximum(.5, expected_changes - 1) / mask.sum(), mask.shape)
    definite_indices = jax.random.randint(key_randint, mask.shape[:-1], 0, mask.shape[-1])
    definite_mask = jax.nn.one_hot(definite_indices, to_change.shape[-1], dtype=jnp.bool_)
    to_change = to_change | definite_mask
    changed = jnp.where(mask & to_change,
                        jax.random.categorical(key_categorical, logit),
                        elite)
    
    key_swap, key_mutations = jax.random.split(key_mutations)
    swap_indices = jax.random.randint(key_swap, (2,), 0, mask.sum())
    swap_from, swap_to = jnp.nonzero(mask, size=len(mask))[0][swap_indices]
    key_swap, key_mutations = jax.random.split(key_mutations)
    do_swap = jax.random.bernoulli(key_swap, p=PROB_SWAP)
    swapped = changed.at[swap_from].set(changed[swap_to]).at[swap_to].set(changed[swap_from])
    changed = jax.lax.cond(do_swap, lambda x: swapped, lambda x: x, changed)
    
    # key_delete, key_mutations = jax.random.split(key_mutations)
    # delete_index = jax.random.randint(key_delete, (), 0, mask.sum())
    # indices = jnp.cumsum(mask)
    # indices_base = jnp.arange(len(changed))
    # deleted = jnp.where(mask, jnp.where(indices > delete_index, indices_base + 1, indices_base), changed)
    # key_delete, key_mutations = jax.random.split(key_mutations)
    # do_delete = jax.random.bernoulli(key_delete, p=0.1)
    # changed = jax.lax.cond(do_delete, lambda x: deleted, lambda x: x, changed)
    
    return changed

# @partial(jax.jit, static_argnames=("key", "candidates", "expected_changes"))
def algo_iteration(elites, vector, key, candidates=64, seed=13, expected_changes=1.5, max_inv_temp=2, topk=128):
    elites = elites.untag("solutions").tag("batch")
    logits, _, _, grads = run_tokens(elites, grad_metric=lambda _l, r, _i: (r[key][{"seq": -1}].untag("embedding") * vector).sum().data_array.mean())
    grads = pz.nx.nmap(lambda x: x >= jax.lax.top_k(x, topk)[0][-1])(grads.untag("vocabulary")).tag("vocabulary")
    logits = logits.untag("batch").tag("elites")

    tempered_samples = pz.nx.nmap(temper)(
        logits.untag("elites", "seq", "vocabulary"),
        pz.nx.wrap(jax.random.split(jax.random.key(seed), candidates), "batch"),
        elites.untag("batch", "seq"),
        grads.untag("batch", "seq", "vocabulary"),
        pz.nx.wrap(jnp.array(max_inv_temp)), pz.nx.wrap(jnp.array(expected_changes))).tag("seq")
    # tempered_samples = sharding_util.name_to_name_device_put(tempered_samples, llama.mesh, dict(batch="dp", seq="sp"))
    _, new_losses, new_resids = run_tokens(tempered_samples)

    new_scores = (new_resids[key][{"seq": -1}].untag("embedding") * vector).sum().astype(new_losses.dtype)
    metrics = pz.nx.nmap(lambda *xs: jnp.stack(xs))(new_losses, new_scores).tag("metrics")
    solution_axes = [k for k in tempered_samples.named_shape.keys() if k != "seq"]
    solutions = tempered_samples.untag(*solution_axes).flatten().tag("solutions").unwrap("solutions", "seq")
    metrics = metrics.untag(*(k for k in solution_axes if k != "seq")).flatten().tag("solutions").unwrap("solutions", "metrics")

    return solutions, metrics


# In[ ]:


from micrlhf.utils.load_sae import get_jb_it_sae
from micrlhf.utils.vector_storage import save_and_upload_vector, download_vector
import jax.numpy as jnp
import random
layer = 12
sae = get_jb_it_sae()
dictionary = sae["W_dec"]
# features = [321, 330, 2079, 5324, 5373, 8361, 8618, 8631, 12017]
# features += [4597, 10681, 11046, 11256, 12701, 15553]
random.seed(3)
features = random.sample(list(range(16384)), 100)
features *= 3
for feature in features:
    vector = dictionary[feature]
    vector = vector / jnp.linalg.norm(vector)
    sae_type = "Residual"

    import random

    rng_seed = random.randint(0, 2**32-1)
    print("Seed:", rng_seed, "Feature:", feature, "SAE type:", sae_type)
    np.random.seed(rng_seed)
    toks_init = tokens_init.copy()
    toks_init[:, optim_mask] = np.random.randint(100, tokenizer.vocab_size, toks_init[:, optim_mask].shape)
    best_metrics = None
    best = tokens_to_array(toks_init).untag("batch").tag("solutions")
    xent_min = 1/20
    xent_max = 20
    weights = jnp.stack((
        -jnp.exp(jnp.linspace(jnp.log(xent_min), jnp.log(xent_max), MAX_ELITES))[::-1],
        jnp.ones(MAX_ELITES),
    ), -1)
    @partial(jax.jit, donate_argnums=(0, 1))
    def combine_solutions(best_metrics, best, metrics, solutions):
        if best_metrics is not None:
            best_metrics = jnp.concatenate((best_metrics, metrics), 0)
            best = pz.nx.nmap(lambda a, b: jnp.concatenate((a, b)))(
                best.untag("solutions"),
                pz.nx.wrap(solutions, "solutions", "seq").untag("solutions")
            ).tag("solutions").unwrap("solutions", "seq")
        else:
            best_metrics = metrics
            best = solutions
        elite_mask = (best_metrics[None, :] * weights[:, None]).sum(-1).argmax(1)
        best_metrics = best_metrics[elite_mask]
        best = pz.nx.wrap(best[elite_mask], "solutions", "seq")
        return best_metrics, best
    try:
        for seed in (bar := trange(750)):
            solutions, metrics = algo_iteration(best, vector, seed=seed, key=f"resid_pre.{layer}")
            best_metrics, best = combine_solutions(best_metrics, best, metrics, solutions)
            m = {}
            for index in range(MAX_ELITES):
                i = index
                m |= {f"decoded.{i}": tokenizer.decode(best[{"solutions": index}].unwrap("seq")).replace("\n", "\\n"),
                    f"loss.{i}": best_metrics[index][0], f"score.{i}": best_metrics[index][1]}
            bar.set_postfix(**m)
    except KeyboardInterrupt:
        pass

    examples = []
    for index in range(MAX_ELITES):
        activation_values = pz.nx.nmap(lambda x: x @ vector)(get_resids_call(llama.inputs.from_basic_segments(best[{"solutions": index}]))[1][layer].value.untag("embedding")).unwrap("seq").tolist()
        bos = tokenizer.encode("")[0]
        tokens = [s[5:] for s in tokenizer.batch_decode([[bos] + [t] for t in best[{"solutions": index}].unwrap("seq").tolist()])]
        examples.append(dict(
            Activations_per_token=activation_values,
            Tokens=tokens,
            Cross_entropy_score=float(best_metrics[index][0]), EPO_metric_score=float(best_metrics[index][1]),
        ))

    import os
    import json
    os.makedirs("data/epo", exist_ok=True)
    feat_id = feature if sae_type not in ('Refusal', 'Sycophancy') else None
    short_metadata = f"{rng_seed}-{layer}-{sae_type}-{feat_id}-{n_x}-{has_end}-{MAX_ELITES}-{xent_min}-{xent_max}"
    run_info = {    
        "SAE_metadata": {
            "Layer": layer,
            "SAE_type": sae_type,
            "Feature_ID": feat_id,
            "Has_end": has_end,
            "N_tokens": n_x,
            "N_elites": MAX_ELITES,
            "Seed": rng_seed,
            "Xent_min": xent_min,
            "Xent_max": xent_max,
        },
        "Examples": examples
    }
    open(f"data/epo/{short_metadata}.json", "w").write(json.dumps(run_info))


# In[ ]:





# In[ ]:




