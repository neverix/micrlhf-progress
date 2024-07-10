import tarfile

import jax
import pandas as pd
from more_itertools import chunked
from penzai import pz
from tqdm.auto import tqdm

from micrlhf.sampling import jit_wrapper, jnp
from micrlhf.scan import sequential_to_scan

combined_prompts = dict(phi="""<|user|>
{}
Choices:
(A) {}
(B) {}
(C) {}
(D) {}<|end|>
<|assistant|>
Answer: ({}""", gemma="""<start_of_turn>user\n
{}
Choices:
(A) {}
(B) {}
(C) {}
(D) {}\n
<start_of_turn>model\n
Answer: ({}""")


class MMLUEval(object):
    def __init__(self, dataset_path: str = "data/mmlu.tar", prompt_format="phi"):
        dataset = []
        with tarfile.open(dataset_path) as data:
            for m in data.getmembers():
                if not m.name.startswith("data/val"):
                    continue
                if not m.name.endswith(".csv"):
                    continue
                df = pd.read_csv(data.extractfile(m))
                for _, r in df.iterrows():
                    dataset.append(combined_prompts[prompt_format].format(*r.tolist()))
        self.dataset = dataset

    def evaluate(self, llama, tokenizer, batch_size=128, verbose=True):
        ps = tokenizer.padding_side
        tokenizer.padding_side = "right"
        accuracies = []
        @jax.jit
        def get_probs(llama, token_array, mask):
            token_array = pz.nx.wrap(token_array, "batch", "seq").untag("batch").tag("batch")
            inputs = llama.inputs.from_basic_segments(token_array)
            logits = llama(inputs)
            probs = pz.nx.nmap(jax.nn.softmax)(logits.untag("vocabulary")).tag("vocabulary")
            mask = pz.nx.wrap(mask, "batch", "seq")
            probs = pz.nx.nmap(lambda p, m, i: p[m.sum() - 2, i[m.sum() - 1]])(
                probs.untag("seq", "vocabulary"), mask.untag("seq"), token_array.untag("seq"))
            return probs.unwrap("batch")
        for batch in chunked((tqdm(self.dataset) if verbose else self.dataset), batch_size):
            # og_batch_size = len(batch)
            batch = batch + [""] * (batch_size - len(batch))
            tokens = tokenizer.batch_encode_plus(batch,
                                                return_tensors="np",
                                                padding="max_length",
                                                truncation=True,
                                                max_length=256,
                                                return_attention_mask=True)
            token_array = jnp.asarray(tokens["input_ids"])
            token_array = jax.device_put(token_array, jax.sharding.NamedSharding(llama.mesh, jax.sharding.PartitionSpec("dp", "sp")))
            mask = jnp.asarray(tokens["attention_mask"])
            mask = jax.device_put(mask, jax.sharding.NamedSharding(llama.mesh, jax.sharding.PartitionSpec("dp", "sp")))
            probs = get_probs(llama, token_array, mask)
            accuracies.extend(probs.tolist())
        tokenizer.padding_side = ps
        return sum(accuracies) / len(accuracies)