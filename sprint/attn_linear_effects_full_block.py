#!/usr/bin/env python
# coding: utf-8

cool_feats = """present_simple_gerund          8446    -> 15554
present_simple_past_perfect    19628   -> 15356
plural_singular                29228   -> 2930
algo_last                      29228   -> 8633
location_country               11459   -> 7967
location_continent             11459   -> 19260
person_profession              26436   -> 18416
football_player_position       19916   -> 9790
present_simple_past_simple     21327   -> 15356
es_en                          31123   -> 5579
fr_en                          31123   -> 16490
it_en                          31123   -> 5579
country_capital                13529   -> 11173
antonyms                       11050   -> 11618
singular_plural                1322    -> 32417
person_language                1132    -> 11172
algo_second                    32115   -> 1878
algo_first                     32115   -> 6756
location_religion              3466    -> 9178
en_fr                          7928    -> 26987
en_it                          7928    -> 26987
location_language              10884   -> 11172
en_es                          99      -> 26987"""
detectors = {}
executors = {}
for line in cool_feats.split("\n"):
    task_name, _, rest = line.partition(" ")
    source, target = [int(x.strip()) for x in rest.split("->")]
    detectors[task_name] = [source]
    executors[task_name] = [target]
print("source", detectors)
print("target", executors)

all_detectors = set(
    [x for sublist in detectors.values() for x in sublist]
)

all_executors = set(
    [x for sublist in executors.values() for x in sublist]
)

# In[1]:


import os
if "models" not in os.listdir("."):
    os.chdir("../..")


# In[2]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
import penzai
import jax_smi
jax_smi.initialise_tracking()
from penzai import pz
# pz.ts.register_as_default()
# pz.ts.register_autovisualize_magic()
# pz.enable_interactive_context()


# In[3]:


# get_ipython().run_line_magic('env', 'JAX_TRACEBACK_FILTERING=off')
import jax
jax.config.update('jax_traceback_filtering', 'off')


# In[4]:


from sprint.icl_sfc_utils import Circuitizer


# In[5]:


from micrlhf.llama import LlamaTransformer
llama = LlamaTransformer.from_pretrained("models/gemma-2b-it.gguf", from_type="gemma", load_eager=True, device_map="tpu:0")


# In[6]:


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("alpindale/gemma-2b")
tokenizer.padding_side = "right"


# In[7]:


from sprint.task_vector_utils import load_tasks, ICLRunner
tasks = load_tasks()


# In[8]:


import json
# detectors = json.load(open("data/detector_heatmap_l11.json"))
# executors = json.load(open("data/executor_heatmap_l12.json"))



# In[9]:


from matplotlib import pyplot as plt
import jax.numpy as jnp
from micrlhf.utils.load_sae import sae_encode, weights_to_resid
from micrlhf.llama import LlamaBlock, LlamaAttention, LlamaInputs
from tqdm.auto import tqdm
import gc


def check_if_single_token(token):
    return len(tokenizer.tokenize(token)) == 1

import os
# get_ipython().system('rm data/attn_out/*.png')
# os.system('rm data/attn_out/*.png')
def plot_attn(task_name):

    task = tasks[task_name]

    print(task_name, len(task))

    # task = {
    #     k:v for k,v in task.items() if check_if_single_token(k) and check_if_single_token(v)
    # }

    print(len(task))

    pairs = list(task.items())

    batch_size = 8
    n_shot=12
    max_seq_len = 128
    seed = 10

    prompt = "Follow the pattern:\n{}"

    runner = ICLRunner(task_name, pairs, batch_size=batch_size, n_shot=n_shot,
                    max_seq_len=max_seq_len, seed=seed,
                    prompt=prompt, use_same_examples=False)

    layers = [11, 12] #, 13, 14]
    circuitizer = Circuitizer(llama, tokenizer, runner, layers, prompt)




    # task_source_resid_features = {11: detectors[task_name]} #["features"]}
    task_source_resid_features = {11: all_detectors} #["features"]}
    # task_source_resid_features = {11: detectors[task_name]}
    # task_target_resid_features = {12: [28800, 16172, 19051,  3925, 22162, 27165, 24640, 26427, 31442,
    #         1425, 25273,  6685, 25966,  5854, 29007, 30363],}#[]}
    # task_target_resid_features = {12: executors[task_name]}#["features"],}#[]}
    task_target_resid_features = {12: all_executors}#["features"],}#[]}
    # task_target_resid_features = {12: executors[task_name]}
    # task_attn_out_features = {11: [4080]}
    task_attn_out_features = {}
    for layer in layers[:-1]:
        resid_sae = circuitizer.get_sae(layer=layer)
        source_resid_features = task_source_resid_features.get(layer, [])
        attn_out_features = task_attn_out_features.get(layer, [])
        target_resid_features = task_target_resid_features.get(layer + 1, [])
        if not source_resid_features or not (attn_out_features or target_resid_features):
            continue

        # qk_n.named_shape
        # out_raw = jnp.einsum("bokp,bkqso->bskqp", v, qk)
        # out_n = pz.nx.wrap(out_raw, "batch", "seq", "kv_heads", "q_rep", "projection")
        # out = attn_layer.attn_value_to_output(out_n)

        next_sae = circuitizer.get_sae(layer=layer+1)
        attn_sae = circuitizer.get_sae(layer=layer, label="attn_out")
        resid_sae = {k: v.astype(jnp.float32) for k, v in resid_sae.items()}
        attn_sae = {k: v.astype(jnp.float32) for k, v in attn_sae.items()}
        next_sae = {k: v.astype(jnp.float32) for k, v in next_sae.items()}
        
        # new_source = []
        # for source_resid_feature in source_resid_features:
        #     if detectors["heatmap"][detectors["task_names"].index(task_name)][detectors["features"].index(source_resid_feature)] <= 0.2:
        #         print("Warning: skipping source feature", source_resid_feature)
        #         continue
        #     new_source.append(source_resid_feature)
        # source_resid_features = new_source
        # new_target = []
        # for target_resid_feature in target_resid_features:
        #     if executors["heatmap"][executors["task_names"].index(task_name)][executors["features"].index(target_resid_feature)] <= 0.2:
        #         print("Warning: skipping target feature", target_resid_feature)
        #         continue
        #     new_target.append(target_resid_feature)
        # target_resid_features = new_target

        # layer = 12
        for source_resid_feature in source_resid_features:
            biggest_feature = circuitizer.ie_resid[layer].mean((0, 1)).argmax().tolist()
            attn_features = circuitizer.ie_attn[layer].mean((0, 1))
            biggest_attn_feature = jnp.argsort(attn_features)[-2]
            # source_resid_feature = biggest_feature
            # attn_out_feature = biggest_attn_feature
            r_pre = circuitizer.resids_pre[layer].astype(jnp.float32)
            r_mid = circuitizer.resids_mid[layer].astype(jnp.float32)
            # attn_out = r_mid - r_pre
            # _, pre_encodings, recon = sae_encode(resid_sae, r_pre)
            # pre_encodings = pre_encodings * jnp.zeros(pre_encodings.shape[-1]).at[source_resid_feature].set(25)
            # err_r = r_pre - recon
            # _, attn_encodings, _ = sae_encode(attn_sae, attn_out)
            # r_other = weights_to_resid(pre_encodings, resid_sae) + err_r
            attn_subblock = circuitizer.llama.select().at_instances_of(LlamaBlock).pick_nth_selected(layer).at_instances_of(pz.nn.Residual).pick_nth_selected(0).get().delta
            attn_layer = attn_subblock.select().at_instances_of(LlamaAttention).pick_nth_selected(0).get()
            attn_ln = attn_subblock.select().at_instances_of(pz.nn.RMSLayerNorm).pick_nth_selected(0).get()




            r_other = r_pre

            _, feature_activation, _ = sae_encode(resid_sae, r_other)
            feature_activation = feature_activation[..., source_resid_feature]
            average_activation = feature_activation.mean()

            if average_activation < 1e-6:
                continue

            direction = resid_sae["W_dec"][source_resid_feature]
            direction = direction / jnp.linalg.norm(direction)
            r_other = r_other - direction * average_activation * 50
            r_other_n = pz.nx.wrap(r_other, "batch", "seq", "embedding")

            attn_input = attn_ln(r_other_n)
            v_n = attn_layer.input_to_value(attn_input)
            # v = v_n.unwrap("batch", "seq", "kv_heads", "projection")
            qk = circuitizer.qk[layer]
            qk_n = pz.nx.wrap(qk, "batch", "kv_heads", "q_rep", "seq", "kv_seq")
            out_n = attn_layer.attn_value_to_output((qk_n, v_n))
        
            mlp_subblock = circuitizer.llama.select().at_instances_of(LlamaBlock).pick_nth_selected(layer).at_instances_of(pz.nn.Residual).pick_nth_selected(1).get()

            out_n = out_n + r_other_n

            out_n = mlp_subblock(out_n)

            out = out_n.unwrap("batch", "seq", "embedding").astype(jnp.float32)

            next_resid_new = out

            
            # out = out_n.unwrap("batch", "seq", "embedding").astype(jnp.float32)
            # _, alt_attn_encodings, _ = sae_encode(attn_sae, out)

            # for attn_out_feature in attn_out_features:
            #     proportions_feature = alt_attn_encodings[..., attn_out_feature] / attn_encodings[..., attn_out_feature]
            #     proportions_feature = jax.nn.relu(jnp.minimum(proportions_feature, 1))
            #     plt.title(f"R {layer} {source_resid_feature} -> A {layer} {attn_out_feature}")
            #     plt.hist(proportions_feature.flatten().tolist(), bins = jnp.linspace(0, 1, 10))
            #     plt.xlabel("Proportion of feature activation")
            #     plt.show()
            
            next_resid = circuitizer.resids_pre[layer + 1].astype(jnp.float32)

            _, target_encodings, _ = sae_encode(next_sae, next_resid)
            # _, alt_target_encodings, _ = sae_encode(next_sae, next_resid + (out - attn_out))
            # _, alt_target_encodings, _ = sae_encode(next_sae, next_resid + (attn_out - out))
            _, alt_target_encodings, _ = sae_encode(next_sae, next_resid_new)
            # _, alt_target_encodings, _ = sae_encode(next_sae, attn_out)

            for target_resid_feature in target_resid_features:
                proportions_feature = alt_target_encodings[..., target_resid_feature] / target_encodings[..., target_resid_feature]
                proportions_feature = 1 - jax.nn.relu(jnp.minimum(1, proportions_feature))
                proportions_feature = proportions_feature[~jnp.isnan(proportions_feature)]
                if not proportions_feature.size:
                    print(f"Max activation for", layer + 1, target_resid_feature, target_encodings[..., target_resid_feature].max())
                    print("Skipping", source_resid_feature, target_resid_feature, "(no activations)")
                    continue
                if proportions_feature.max() < 0.1:
                    print("Skipping", source_resid_feature, target_resid_feature, "(no high proportion)")
                    continue
                # plt.title(f"Task {task_name}: R {layer} {source_resid_feature} -> A {layer} -> R {layer+1} {target_resid_feature}")
                # plt.hist(proportions_feature.flatten().tolist(), bins = jnp.linspace(0, 1, 10))
                # # try:
                # #     plt.hist(proportions_feature.flatten().tolist())#, bins = jnp.linspace(0, 1, 10))
                # # except ValueError:
                # #     plt.close()
                # #     continue
                # plt.xlabel("Proportion of feature activation")
                # plt.savefig(f"data/attn_out/{task_name}_{layer}_{source_resid_feature}_{layer}_{layer+1}_{target_resid_feature}.png")
                # plt.close()

                with open("micrlhf-progress/attn_results_4.jsonl", "a") as f:
                    f.write(json.dumps({
                        "task": task_name,
                        "source": source_resid_feature,
                        "target": target_resid_feature,
                        "proportions": proportions_feature.flatten().tolist()
                    }) + "\n")

# for detector_tni, task_name in enumerate(tqdm(detectors["task_names"])):
# # task_name = "antonyms"
# # task_name = "location_language"
    # plot_attn(task_name)
import sys
plot_attn(sys.argv[1])
#     gc.collect()


# # In[ ]:





# # In[ ]:


# # jax.lax.top_k((circuitizer.ie_resid[12] * circuitizer.masks["arrow"][..., None]).mean((0, 1)), 16)
# jax.lax.top_k((target_encodings * circuitizer.masks["arrow"][..., None]).mean((0, 1)), 16)


# # In[ ]:




