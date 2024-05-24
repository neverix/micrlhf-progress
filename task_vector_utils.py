import random
import os
import subprocess
import glob
import json
from typing import List
import jax
import jax.numpy as jnp
import numpy as np
from micrlhf.utils.activation_manipulation import add_vector


def generate_algorithmic_tasks(seed = 0, n_examples=300, max_len=10, max_value=100):
    generator = random.Random(seed)
    tasks = {}

    tasks["algo_max"] = {}
    for _ in range(n_examples):
        length = generator.randint(1, max_len)
        a = [generator.randint(0, max_value) for _ in range(length)]
        tasks["algo_max"][f"{a}"] = f"{max(a)}"

    tasks["algo_min"] = {}
    for _ in range(n_examples):
        length = generator.randint(1, max_len)
        a = [generator.randint(0, max_value) for _ in range(length)]
        tasks["algo_min"][f"{a}"] = f"{min(a)}"

    tasks["algo_last"] = {}
    for _ in range(n_examples):
        length = generator.randint(1, max_len)
        a = [generator.randint(0, max_value) for _ in range(length)]
        tasks["algo_last"][f"{a}"] = f"{a[-1]}"
    
    tasks["algo_first"] = {}
    for _ in range(n_examples):
        length = generator.randint(1, max_len)
        a = [generator.randint(0, max_value) for _ in range(length)]
        tasks["algo_first"][f"{a}"] = f"{a[0]}"

    tasks["algo_sum"] = {}
    for _ in range(n_examples):
        length = generator.randint(1, max_len)
        a = [generator.randint(0, max_value) for _ in range(length)]
        tasks["algo_sum"][f"{a}"] = f"{sum(a)}"
    
    tasks["algo_most_common"] = {}
    for _ in range(n_examples):
        length = generator.randint(1, max_len)
        a = [generator.randint(0, max_value) for _ in range(length)]
        tasks["algo_most_common"][f"{a}"] = f"{max(set(a), key=a.count)}"

    return tasks


def load_tasks():
    subprocess.run(["git", "clone", "https://github.com/roeehendel/icl_task_vectors data/itv"])
    tasks = {}
    for g in glob.glob("data/itv/data/**/*.json"):
        tasks[os.path.basename(g).partition(".")[0]] = json.load(open(g))

    tasks.update(generate_algorithmic_tasks())

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

def logprob_loss(logits, tokens, sep=1599, n_first=None, shift=None):
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

    # print(mask[:, -5:])

    
    logits = logits * mask

    return -logits.sum(axis=-1).mean(axis=-1)

def make_act_adder(llama, tv, tokens, layer, length=1, sep=1599, shift=0):
    mask = tokens == sep

    positions = jnp.argwhere(mask)[:, -1]
    positions = jnp.column_stack(
        tuple(
            positions + i + shift
            for i in range(length)
        )
    )

    add_act = add_vector(llama, tv, layer, 1, position = positions)

    return add_act

def get_tv(resids, tokens, sep=1599, shift=None):
    mask = tokens == sep
    
    if shift is not None:
        mask = jnp.roll(mask, shift, axis=-1)

    tv = resids[mask]

    return tv.mean(axis=0)


class ICLRunner:
    def __init__(self, task: str, pairs: list[list[str]], seed=0, batch_size=20, n_shot=5, max_seq_len=128):
        self.task = task
        self.pairs = pairs
        self.seed = seed
        self.batch_size = batch_size
        self.n_shot = n_shot
        self.max_seq_len = max_seq_len
        
        self.prepend_space = task.startswith("algo")

        self.gen = random.Random(seed)

        self.train_pairs = [self.gen.sample(pairs, k=n_shot) for _ in range(batch_size)]
        self.eval_pairs = [self.gen.sample(pairs, k=1) for _ in range(batch_size)]

        self.prompt = "<|user|>\nFollow the pattern:\n{}"

    def get_prompt(self, pairs):
        return self.prompt.format("\n".join([f"{x} -> {y}" for x, y in pairs]))

    def get_tokens(self, pairs, tokenizer):
        prompts = [self.get_prompt(p) for p in pairs]
        
        tokenized = tokenizer(prompts, padding="longest", return_tensors="np")

        assert tokenized["input_ids"].shape[1] <= self.max_seq_len, "Prompt too long for model."

        return tokenized
    
