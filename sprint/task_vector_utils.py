import dataclasses
from functools import partial
import random
import os
import subprocess
import glob
import json
from typing import List
import jax
import jax.numpy as jnp
import numpy as np
import optax
from penzai import pz
from tqdm.auto import trange, tqdm
from micrlhf.llama import LlamaBlock
from micrlhf.utils.activation_manipulation import add_vector
from micrlhf.utils.load_sae import get_sae, sae_encode, weights_to_resid
from micrlhf.sampling import jit_wrapper



def generate_algorithmic_tasks(seed = 0, n_examples=300, length=4, max_value=100):
    generator = random.Random(seed)
    tasks = {}

    tasks["algo_max"] = {}
    for _ in range(n_examples):
        a = [generator.randint(0, max_value) for _ in range(length)]
        tasks["algo_max"][f"{a}"] = f"{max(a)}"

    tasks["algo_min"] = {}
    for _ in range(n_examples):
        a = [generator.randint(0, max_value) for _ in range(length)]
        tasks["algo_min"][f"{a}"] = f"{min(a)}"

    tasks["algo_last"] = {}
    for _ in range(n_examples):
        a = [generator.randint(0, max_value) for _ in range(length)]
        tasks["algo_last"][f"{a}"] = f"{a[-1]}"
    
    tasks["algo_first"] = {}
    for _ in range(n_examples):
        a = [generator.randint(0, max_value) for _ in range(length)]
        tasks["algo_first"][f"{a}"] = f"{a[0]}"

    # tasks["algo_sum"] = {}
    # for _ in range(n_examples):
    #     length = generator.randint(min_len, max_len)
    #     a = [generator.randint(0, max_value) for _ in range(length)]
    #     tasks["algo_sum"][f"{a}"] = f"{sum(a)}"
    
    # tasks["algo_most_common"] = {}
    # for _ in range(n_examples):
    #     length = generator.randint(min_len, max_len)
    #     a = [generator.randint(0, max_value) for _ in range(length)]
    #     tasks["algo_most_common"][f"{a}"] = f"{max(set(a), key=a.count)}"

    return tasks

def generate_algorithmic_tasks_words(items, seed = 0, n_examples=300, min_length=4, max_length=4):
    generator = random.Random(seed)
    tasks = {}


    tasks["algo_last"] = {}
    for _ in range(n_examples):
        length = generator.randint(min_length, max_length)
        a = generator.sample(items, length)
        tasks["algo_last"][", ".join(a)] = f"{a[-1]}"
    
    tasks["algo_first"] = {}
    for _ in range(n_examples):
        length = generator.randint(min_length, max_length)
        a = generator.sample(items, length)
        tasks["algo_first"][", ".join(a)] = f"{a[0]}"


    tasks["algo_second"] = {}
    for _ in range(n_examples):
        length = generator.randint(min_length, max_length)
        a = generator.sample(items, length)
        tasks["algo_second"][", ".join(a)] = f"{a[1]}"

    # tasks["algo_sum"] = {}
    # for _ in range(n_examples):
    #     length = generator.randint(min_len, max_len)
    #     a = [generator.randint(0, max_value) for _ in range(length)]
    #     tasks["algo_sum"][f"{a}"] = f"{sum(a)}"
    
    # tasks["algo_most_common"] = {}
    # for _ in range(n_examples):
    #     length = generator.randint(min_len, max_len)
    #     a = [generator.randint(0, max_value) for _ in range(length)]
    #     tasks["algo_most_common"][f"{a}"] = f"{max(set(a), key=a.count)}"

    return tasks


def load_tasks():
    subprocess.run(["git", "clone", "https://github.com/roeehendel/icl_task_vectors", "data/itv"])
    tasks = {}
    for g in glob.glob("data/itv/data/**/*.json"):
        tasks[os.path.basename(g).partition(".")[0]] = json.load(open(g))

    # tasks.update(generate_algorithmic_tasks())

    items = list(tasks["en_es"].keys())

    tasks.update(generate_algorithmic_tasks_words(items))

    return tasks

class ICLSequence:
    '''
    Class to store a single antonym sequence.

    Uses the default template "Q: {x}\nA: {y}" (with separate pairs split by "\n\n").
    '''
    def __init__(self, word_pairs: List[List[str]], prepend_space=False):
        self.word_pairs = word_pairs
        self.x, self.y = zip(*word_pairs)
        self.prepend_space = prepend_space

    def __len__(self):
        return len(self.word_pairs)

    def __getitem__(self, idx: int):
        return self.word_pairs[idx]

    # def prompt(self):
    #     '''Returns the prompt, which contains all but the second element in the last word pair.'''
    #     p = "\n\n".join([f"Q: {x}\nA: {y}" for x, y in self.word_pairs])
    #     return p[:-len(self.completion())]

    def prompt(self):
        '''Returns the prompt, which contains all but the second element in the last word pair.'''
        p = ", ".join([f"{x} -> {y}" for x, y in self.word_pairs])

        if self.prepend_space:
            return " " + p[:-len(self.completion())]
        return p[:-len(self.completion()) -1]

    def completion(self):
        '''Returns the second element in the last word pair (with padded space).'''
        return "" + self.y[-1]

    def __str__(self):
        '''Prints a readable string representation of the prompt & completion (indep of template).'''
        return f"{', '.join([f'({x}, {y})' for x, y in self[:-1]])}, {self.x[-1]} ->".strip(", ")

class ICLDataset:
    '''
    Dataset to create antonym pair prompts, in ICL task format. We use random seeds for consistency
    between the corrupted and clean datasets.

    Inputs:
        word_pairs:
            list of ICL task, e.g. [["old", "young"], ["top", "bottom"], ...] for the antonym task
        size:
            number of prompts to generate
        n_prepended:
            number of antonym pairs before the single-word ICL task
        bidirectional:
            if True, then we also consider the reversed antonym pairs
        corrupted:
            if True, then the second word in each pair is replaced with a random word
        seed:
            random seed, for consistency & reproducibility
    '''

    def __init__(
        self,
        word_pairs: List[List[str]],
        size: int,
        n_prepended: int,
        bidirectional: bool = True,
        seed: int = 0,
        corrupted: bool = False,
        prepend_space: bool = False
    ):
        assert n_prepended+1 <= len(word_pairs), "Not enough antonym pairs in dataset to create prompt."

        self.word_pairs = word_pairs
        self.word_list = [word for word_pair in word_pairs for word in word_pair]
        self.size = size
        self.n_prepended = n_prepended
        self.bidirectional = bidirectional
        self.corrupted = corrupted
        self.seed = seed
        self.prepend_space = prepend_space

        self.seqs = []
        self.prompts = []
        self.completions = []

        # Generate the dataset (by choosing random antonym pairs, and constructing `ICLSequence` objects)
        for n in range(size):
            np.random.seed(seed + n)
            random_pairs = np.random.choice(len(self.word_pairs), n_prepended+1, replace=False)
            random_orders = np.random.choice([1, -1], n_prepended+1)
            if not(bidirectional): random_orders[:] = 1
            word_pairs = [self.word_pairs[pair][::order] for pair, order in zip(random_pairs, random_orders)]
            if corrupted:
                for i in range(len(word_pairs) - 1):
                    word_pairs[i][1] = np.random.choice(self.word_list)
            seq = ICLSequence(word_pairs, prepend_space=self.prepend_space)

            self.seqs.append(seq)
            self.prompts.append(seq.prompt())
            self.completions.append(seq.completion())

    def create_corrupted_dataset(self):
        '''Creates a corrupted version of the dataset (with same random seed).'''
        return ICLDataset(self.word_pairs, self.size, self.n_prepended, self.bidirectional, corrupted=True, seed=self.seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        return self.seqs[idx]

def logprob_loss(logits, tokens, sep=1599, pad_token=32000, n_first=None, shift=None):
    logits = jax.nn.log_softmax(logits)

    logits = logits[:, :-1]

    # print(
    #     logits.argmax(axis=-1)
    # )

    logits = jnp.take_along_axis(logits, tokens[:, 1:, None], axis=-1).squeeze(-1)

    mask = tokens[:, 1:] == sep
    mask = jnp.cumsum(mask[:, ::-1], axis=-1)[:, ::-1] > 0
    mask = jnp.logical_not(mask)

    if shift is not None:
        rolled_mask = jnp.roll(mask, shift, axis=-1)
        mask = jnp.logical_and(mask, rolled_mask)

    # print(mask[:, -5:])
    
    if n_first is not None:
        rolled_mask = jnp.roll(mask, n_first, axis=-1)
        mask = jnp.logical_and(mask, jnp.logical_not(rolled_mask))

    mask = jnp.logical_and(mask, tokens[:, 1:] != pad_token)

    logits = logits * mask

    return -logits.sum(axis=-1).mean(axis=-1)

def task_vector_mask(tokens, sep=1599, shift=None):
    mask = tokens == sep
    
    if shift is not None:
        mask = jnp.roll(mask, shift, axis=-1)

    return mask

def make_act_adder(llama, tv, tokens, layer, length=1, sep=1599, shift=0, scale=1):
    mask = tokens == sep

    col_indices = jnp.arange(mask.shape[1])

    col_indices_broadcasted = mask * col_indices

    sorted_indices = jnp.sort(col_indices_broadcasted, axis=1, descending=True)

    k = jnp.sum(mask[0]).astype(int)

    positions = sorted_indices[:, :k]

    positions = jnp.column_stack(
        tuple(
            positions + i + shift
            for i in range(length)
        )
    )

    add_act = add_vector(llama, tv, layer, scale, position = positions)

    return add_act

def make_act_adder_detector(llama, tv, tokens, layer, prompt_length, length=1, newline=108, shift=0, scale=1):
    mask = tokens == newline
    mask = jnp.roll(mask, -1, axis=-1)
    mask = mask.at[:, :prompt_length].set(False)

    col_indices = jnp.arange(mask.shape[1])

    col_indices_broadcasted = mask * col_indices

    sorted_indices = jnp.sort(col_indices_broadcasted, axis=1, descending=True)

    k = jnp.sum(mask[0]).astype(int)

    positions = sorted_indices[:, :k]

    positions = jnp.column_stack(
        tuple(
            positions + i + shift
            for i in range(length)
        )
    )

    add_act = add_vector(llama, tv, layer, scale, position = positions)

    return add_act
    

def get_tv(resids, tokens, sep=1599, shift=None):
    mask = tokens == sep
    
    if shift is not None:
        mask = jnp.roll(mask, shift, axis=-1)

    tv = resids[mask]

    return tv.mean(axis=0)

def get_tv_detector(resids, tokens, prompt_length, newline=108, shift=None):
    mask = tokens == newline
    mask = jnp.roll(mask, -1, axis=-1)
    mask = mask.at[:, :prompt_length].set(False)

    tv = resids[mask]

    return tv.mean(axis=0)


class ICLRunner:
    def __init__(self, task: str, pairs: list[list[str]], seed=0, batch_size=20, n_shot=5, max_seq_len=128, prompt=None, eval_size=2, use_same_examples=False, use_same_target=False, vector_type="executor"):
        self.task = task
        self.pairs = pairs
        self.seed = seed
        self.batch_size = batch_size
        self.eval_batch_size = batch_size * eval_size
        self.n_shot = n_shot
        self.max_seq_len = max_seq_len
        self.use_same_examples = use_same_examples
        self.use_same_target = use_same_target
        self.prepend_space = False

        self.gen = random.Random(seed)
        if self.use_same_examples:
            examples = self.gen.sample(pairs, k=n_shot-1)
            self.train_pairs = [examples + [self.gen.sample(pairs, k=1)[0]] for _ in range(batch_size)]

        elif self.use_same_target:
            target = self.gen.sample(pairs, k=1)
            self.train_pairs = [self.gen.sample(pairs, k=n_shot-1) + target for _ in range(batch_size)]
        else:
            self.train_pairs = [self.gen.sample(pairs, k=n_shot) for _ in range(batch_size)]

        # if vector_type == "detector":
        #     self.train_pairs = [
        #         ("X -> Y") + x for x in self.train_pairs
        #     ]
        self.eval_pairs = [self.gen.sample(pairs, k=1) for _ in range(self.eval_batch_size)]

        if vector_type == "detector":
            self.eval_pairs = [
                [("X", "Y")] + x for x in self.eval_pairs
            ]

        if prompt is None:
            self.prompt = "<|user|>\nFollow the pattern:\n{}"
        else:
            self.prompt = prompt

    def get_prompt(self, pairs):
        return self.prompt.format("\n".join([f"{x} -> {y}" for x, y in pairs]))

    def get_tokens(self, pairs, tokenizer):
        prompts = [self.get_prompt(p) for p in pairs]
        
        # tokenized = tokenizer(prompts, padding="longest", return_tensors="np")
        tokenized = tokenizer(prompts, padding="max_length", return_tensors="np", max_length=self.max_seq_len, truncation=False)

        assert tokenized["input_ids"].shape[1] <= self.max_seq_len, "Prompt too long for model."

        return tokenized

def tokenized_to_inputs(input_ids, attention_mask, llama):
    token_array = jnp.asarray(input_ids)
    token_array = jax.device_put(token_array, jax.sharding.NamedSharding(llama.mesh, jax.sharding.PartitionSpec("dp", "sp")))
    token_array = pz.nx.wrap(token_array, "batch", "seq").untag("batch").tag("batch")

    mask_array = jnp.asarray(attention_mask, dtype=jnp.bool)
    mask_array = jax.device_put(mask_array, jax.sharding.NamedSharding(llama.mesh, jax.sharding.PartitionSpec("dp", "sp")))
    mask_array = pz.nx.wrap(mask_array, "batch", "seq").untag("batch").tag("batch")

    inputs = llama.inputs.from_basic_segments(token_array)
    return inputs

jittify = lambda x: partial(jax.jit(lambda lr, *args, **kwargs: lr(*args, **kwargs)[1][0].value), x)

def make_get_resids(llama, layer_target):
    get_resids = llama.select().at_instances_of(LlamaBlock).pick_nth_selected(layer_target
                                                                              ).apply(lambda x:
        pz.nn.Sequential([
            pz.de.TellIntermediate.from_config(tag=f"resid_pre"),
            x
        ])
    )
    get_resids = pz.de.CollectingSideOutputs.handling(get_resids, tag_predicate=lambda x: x.startswith("resid_pre"))
    return get_resids


class FeatureSearch:
    def __init__(self, task, pairs, target_layer, llama, tokenizer, batch_size=32, n_shot=20, early_stopping_steps=50, 
                 max_seq_len=256, iterations=2000, seed=9, l1_coeff=2e-2, lr=1e-2, sep=1599, newline=108, pad_token=32000,
                 init_w=0.5, sae_v=4, n_first=3, picked_features=None, sae=None, prompt=None, feature_type="executor", n_batches=1):
        self.task = task
        self.target_layer = target_layer
        self.sae_v = sae_v
        if sae is None:
            self.sae = get_sae(target_layer, sae_v)
        else:
            self.sae = sae
        self.seed = seed
        self.early_stopping_steps = early_stopping_steps
        self.iterations = iterations
        self.l1_coeff = l1_coeff
        self.lr = lr
        self.init_w = init_w
        self.n_shot = n_shot
        self.batch_size = batch_size
        self.n_first = n_first
        self.picked_features = picked_features
        self.llama = llama
        self.tokenizer = tokenizer
        self.sep = sep
        self.newline = newline
        self.pad_token = pad_token
        self.prompt = prompt
        self.feature_type = feature_type
        self.n_batches = n_batches

        self.prompt_length = tokenizer(self.prompt, return_tensors="np")["input_ids"].shape[1]

        self.runner = ICLRunner(task, pairs, batch_size=batch_size*self.n_batches, n_shot=n_shot, max_seq_len=max_seq_len, seed=seed, prompt=self.prompt, eval_size=1, vector_type=self.feature_type)
        
        self.train_inputs = tokenized_to_inputs(
            **self.runner.get_tokens(self.runner.train_pairs, tokenizer), llama=llama
        )

        self.eval_inputs = [
            tokenized_to_inputs(
                **self.runner.get_tokens(self.runner.eval_pairs[i*self.batch_size:(i+1)*self.batch_size], tokenizer), llama=llama
            ) for i in range(self.n_batches)
        ]

        # self.eval_tokens = self.runner.get_tokens(self.runner.eval_pairs, tokenizer)["input_ids"]

        self.eval_tokens = [
            self.runner.get_tokens(self.runner.eval_pairs[i*self.batch_size:(i+1)*self.batch_size], tokenizer)["input_ids"] for i in range(self.n_batches)
        ]

        self.initial_resids = [
            self.get_initial_resids(inputs) for inputs in self.eval_inputs
        ]

        self.lwg = jax.value_and_grad(self.get_loss, has_aux=True)
        self.taker = self.make_taker()
    
    def get_initial_resids(self, inputs):
        get_resids_initial = make_get_resids(self.llama, self.target_layer)
        get_resids_initial = jittify(get_resids_initial)

        initial_resids = get_resids_initial(inputs)
        return initial_resids

    def make_taker(self):
        taker = jit_wrapper.Jitted(self.llama.select().at_instances_of(LlamaBlock).apply_with_selected_index(
            lambda i, x: x if i >= self.target_layer else pz.nn.Identity()
        ).select().at_instances_of(pz.nn.EmbeddingLookup).apply(lambda _: pz.nn.Identity())
                        .select().at_instances_of(pz.nn.ConstantRescale).pick_nth_selected(0).apply(lambda _: pz.nn.Identity()))

        return taker
    
    def get_loss(self, weights):
        _, weights, recon = sae_encode(self.sae, None, pre_relu=weights)

        # weights = jax.nn.relu(weights)

        # recon = weights_to_resid(weights, self.sae)

        recon = recon.astype('bfloat16')
        
        loss = 0

        for inputs, resids, tokens in zip(self.eval_inputs, self.initial_resids, self.eval_tokens):
            if self.feature_type == "executor":
                mask = tokens == self.sep
            else:
                mask = tokens == self.newline
                mask = jnp.roll(mask, -1, axis=-1)
                mask = mask.at[:, :self.prompt_length].set(False)

            positions = jnp.argwhere(mask)[:, -1]
            
            resids = resids.unwrap("batch", "seq", "embedding")
            modified = jax.vmap(lambda a, b: a.at[b].add(recon))(
                resids, positions
            )
            modified = pz.nx.wrap(modified, "batch", "seq", "embedding")
        
            inputs = dataclasses.replace(inputs, tokens=modified)
            logits = self.taker(inputs).unwrap("batch", "seq", "vocabulary")
            loss += logprob_loss(logits, tokens, n_first=self.n_first, sep=self.sep, pad_token=self.pad_token)

        loss /= self.n_batches

        # self.l1_coeff *= 1.002

        return loss + self.l1_coeff * jnp.linalg.norm(weights, ord=1), (int((weights > self.sae.get("threshold", 0)).sum()), loss)

    def train_step(self, weights, opt_state, optimizer):
        (loss, (l0, loss_)), grad = self.lwg(weights)

        updates, opt_state = optimizer.update(grad, opt_state, weights)
        weights = optax.apply_updates(weights, updates)

        return loss, weights, opt_state, dict(l0=l0, loss=loss_)

    def create_optimizer(self):
        optimizer = optax.chain(
            optax.adam(self.lr),
            optax.zero_nans(),
        )

        return optimizer

    def find_weights(self):
        if isinstance(self.init_w, jnp.ndarray):
            weights = self.init_w
        elif self.picked_features is not None:
            weights = jnp.zeros(self.sae["W_dec"].shape[0])
            weights = weights.at[self.picked_features].set(self.init_w)
        else:
            weights = jnp.ones(self.sae["W_dec"].shape[0]) * self.init_w
        optimizer = self.create_optimizer()
        opt_state = optimizer.init(weights)

        min_loss = 1e9
        early_stopping_counter = 0

        for step_n in (bar := trange(self.iterations)):
            loss, weights, opt_state, metrics = self.train_step(weights, opt_state, optimizer)
            metrics["step"] = step_n

            if metrics["l0"] < min_loss:
                min_loss = metrics["l0"]
                early_stopping_counter = 0

            tk = jax.lax.top_k(weights, 2)

            bar.set_postfix(loss_optim=loss, **metrics, top=tk[1][0], top_diff=(tk[0][0] - tk[0][1]) / tk[0][0])

            early_stopping_counter += 1
            if early_stopping_counter > self.early_stopping_steps:
                break

        return weights, metrics
    
    def check_feature(self, feature, scale):
        steering_vector = self.sae["W_dec"][feature] * scale
        steering_vector =  steering_vector.astype('bfloat16')

        act_add = make_act_adder(
            self.llama, steering_vector, self.eval_tokens, self.target_layer, sep=self.sep
        )

        logits = act_add(self.eval_inputs).unwrap("batch", "seq", "vocabulary")

        return logprob_loss(logits, self.eval_tokens, n_first=self.n_first, sep=self.sep, pad_token=self.pad_token)

    def check_features(self, features, scale):
        losses = jnp.hstack([self.check_feature(feature, scale) for feature in tqdm(features)])

        return features[losses.argmin()], losses.min(), losses.mean(), losses

    