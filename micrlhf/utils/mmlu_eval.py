import tarfile

import jax
import pandas as pd
from penzai import pz
from more_itertools import chunked
from tqdm.auto import tqdm

from micrlhf.sampling import jit_wrapper, jnp
from micrlhf.scan import sequential_to_scan

combined_prompt = """<|user|>
{}
Choices:
(A) {}
(B) {}
(C) {}
(D) {}<|end|>
<|assistant|>
Answer: ({}"""


class MMLUEval(object):
    def __init__(self, dataset_path: str = "data/mmlu.tar"):
        dataset = []
        with tarfile.open(dataset_path) as data:
            for m in data.getmembers():
                if not m.name.startswith("data/val"):
                    continue
                if not m.name.endswith(".csv"):
                    continue
                df = pd.read_csv(data.extractfile(m))
                for _, r in df.iterrows():
                    dataset.append(combined_prompt.format(*r.tolist()))
        self.dataset = dataset

    def evaluate(self, llama, tokenizer, batch_size=128):
        ps = tokenizer.padding_side
        tokenizer.padding_side = "right"
        llama_call = jit_wrapper.Jitted(sequential_to_scan(llama))
        accuracies = []
        for batch in chunked(tqdm(self.dataset), batch_size):
            og_batch_size = len(batch)
            batch = batch + [""] * (batch_size - len(batch))
            tokens = tokenizer.batch_encode_plus(batch,
                                                return_tensors="np",
                                                padding="max_length",
                                                truncation=True,
                                                max_length=256,
                                                return_attention_mask=True)
            token_array = jnp.asarray(tokens["input_ids"])
            token_array = jax.device_put(token_array, jax.sharding.NamedSharding(llama.mesh, jax.sharding.PartitionSpec("dp", "sp")))
            token_array = pz.nx.wrap(token_array, "batch", "seq").untag("batch").tag("batch")
            inputs = llama.inputs.from_basic_segments(token_array)
            logits = llama_call(inputs)
            probs = pz.nx.nmap(jax.nn.softmax)(logits.untag("vocabulary")).tag("vocabulary")
            mask = pz.nx.wrap(jnp.asarray(tokens["attention_mask"]), "batch", "seq")
            probs = pz.nx.nmap(lambda p, m, i: p[m.sum() - 2, i[m.sum() - 1]])(
                probs.untag("seq", "vocabulary"), mask.untag("seq"), token_array.untag("seq"))
            accuracies.extend(probs.unwrap("batch").tolist())
        tokenizer.padding_side = ps
        return sum(accuracies) / len(accuracies)