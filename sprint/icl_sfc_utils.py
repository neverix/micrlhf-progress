import jax
import equinox as eqx
import jax.numpy as jnp
from micrlhf.llama import LlamaBlock, LlamaAttention, LlamaInputs
from micrlhf.utils.activation_manipulation import ActivationAddition, wrap_vector
from functools import partial
import jax.numpy as jnp
from collections import OrderedDict
from penzai import pz
import numpy as np
import jax

from micrlhf.llama import LlamaTransformer
from micrlhf.utils.load_sae import get_nev_it_sae_suite, sae_encode_gated, weights_to_resid, resids_to_weights
from typing import List, Callable, Dict
from transformers import AutoTokenizer
from sprint.task_vector_utils import load_tasks, ICLRunner
from tqdm.auto import tqdm, trange


def logprob_loss(logits, tokens, sep=1599, pad_token=32000, n_first=None, shift=None, use_softmax=False):
    if use_softmax:
        logits = jax.nn.log_softmax(logits)
    
    logits = logits[:, :-1]

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

    return logits.sum(axis=-1).mean(axis=-1)


def load_saes(layers):
    saes = {}
    for layer in tqdm(layers):
        try:
            saes[(layer, "attn_out")] = get_nev_it_sae_suite(layer=layer, label="attn_out")
        except KeyError:
            pass
        try:
            saes[(layer, "resid")] = get_nev_it_sae_suite(layer=layer, label="residual")
        except KeyError:
            pass
        try:
            saes[(layer, "transcoder")] = get_nev_it_sae_suite(layer=layer, label="transcoder")
        except KeyError:
            pass
    return saes

def sfc_simple(grad, resid, target, sae):
    pre_relu, post_relu, recon = sae_encode_gated(sae, resid)

    post_relu = post_relu.astype(jnp.float32)
    error = target - recon
    f = partial(weights_to_resid, sae=sae)


    grad = grad.astype(jnp.float32)
    sae_grad, = jax.vjp(f, post_relu)[1](grad,)
    indirect_effects = sae_grad * post_relu
    indirect_effects_error = jnp.einsum("...f, ...f -> ...", grad, error)
    return indirect_effects, indirect_effects_error, sae_grad, error


sep = 3978
pad = 0

def metric_fn(logits, resids, tokens, use_softmax=False):
    return logprob_loss(logits, tokens, sep=sep, pad_token=pad, n_first=2, use_softmax=use_softmax)



from micrlhf.llama import LlamaMLP
from typing import Dict, List
import dataclasses


@pz.pytree_dataclass
class AblatedModule(pz.Layer):
    module: pz.Layer
    sae: dict
    keep_features: Dict[str, jax.typing.ArrayLike]
    masks: Dict[str, jax.typing.ArrayLike] = dataclasses.field(default_factory=dict)

    @classmethod
    def wrap(self, sae, ablated_features, masks, module=None):
        if module is None:
            module = pz.nn.Identity()
        return AblatedModule(module, sae, ablated_features, masks)

    def __call__(self, x):
        inp = x.unwrap("batch", "seq", "embedding")
        out = self.module(x)
        result = 0
        out = out.unwrap("batch", "seq", "embedding")
        for mask, mask_values in self.masks.items():
            _, _, recon = sae_encode_gated(self.sae, inp)
            error = out - recon
            _, _, recon_ablated = sae_encode_gated(self.sae, inp, keep_features=self.keep_features[mask])
            res = recon_ablated + error
            res = res * mask_values[..., None]
            result = result + res
        return pz.nx.wrap(result, "batch", "seq", "embedding")

class Circuitizer(eqx.Module):
    saes: dict
    llama: LlamaTransformer
    layers: List[int]
    masks: Dict[str, jax.typing.ArrayLike]
    get_rms_block: Callable
    train_tokens: jax.typing.ArrayLike
    tokens_wrapped: pz.nx.NamedArrayBase
    llama_inputs: LlamaInputs
    ie_attn: Dict[int, jax.typing.ArrayLike]
    sae_grads_attn: Dict[int, jax.typing.ArrayLike]
    ie_resid: Dict[int, jax.typing.ArrayLike]
    sae_grads_resid: Dict[int, jax.typing.ArrayLike]
    ie_transcoder: Dict[int, jax.typing.ArrayLike]
    sae_grads_transcoder: Dict[int, jax.typing.ArrayLike]

    ie_error_attn: Dict[int, jax.typing.ArrayLike]
    ie_error_resid: Dict[int, jax.typing.ArrayLike]
    ie_error_transcoder: Dict[int, jax.typing.ArrayLike]

    sae_error_attn: Dict[int, jax.typing.ArrayLike]
    sae_error_resid: Dict[int, jax.typing.ArrayLike]
    sae_error_transcoder: Dict[int, jax.typing.ArrayLike]

    metric_value: jax.typing.ArrayLike
    resids_pre: List[jax.typing.ArrayLike]
    resids_mid: List[jax.typing.ArrayLike]
    qk: List[jax.typing.ArrayLike]
    grad_pre: List[jax.typing.ArrayLike]
    grad_mid: List[jax.typing.ArrayLike]
    mlp_rms: List[jax.typing.ArrayLike]
    

    def __init__(self, llama: LlamaTransformer, tokenizer: AutoTokenizer, runner: ICLRunner, layers: List[int], prompt: str):
        self.llama = llama
        self.layers = layers

        self.train_tokens = runner.get_tokens(
            runner.train_pairs, tokenizer
        )["input_ids"]

        self.get_rms_block = lambda layer, resid_index: (
            self.llama.select()
            .at_instances_of(LlamaBlock).pick_nth_selected(layer)
            .at_instances_of(pz.nn.Residual).pick_nth_selected(resid_index)
            .at_instances_of(pz.nn.RMSLayerNorm).pick_nth_selected(0)
            ).get()


        self.tokens_wrapped = pz.nx.wrap(self.train_tokens, "batch", "seq")
        self.llama_inputs = llama.inputs.from_basic_segments(self.tokens_wrapped)

        print("Setting up masks...")
        prompt_length = len(tokenizer.tokenize(prompt))
        periods = ["input", "arrow", "output", "newline"]
        self.masks = OrderedDict([
            ("prompt", jnp.zeros_like(self.train_tokens).at[:, :prompt_length].set(1).astype(bool)),
            *[
                (period, jnp.zeros_like(self.train_tokens).at[:, prompt_length+i::len(periods)].set(1).astype(bool) * (self.train_tokens != pad)) for i, period in enumerate(periods)
            ]
        ])

        print("Running metrics...")
        self.run_metrics()
        print("Setting up RMS...")
        self.mlp_rms = [self.get_rms_block(layer, 1) for layer in trange(llama.config.num_layers)]
        print("Loading SAEs...")
        self.saes = load_saes(self.layers)
        print("Running node IEs...")
        self.run_node_ies()

    def get_sae(self, layer, label="resid"):
        return self.saes[(layer, label)]


    def run_metrics(self):
        metric_value, resids_pre, resids_mid, qk, grad_pre, grad_mid = self.get_metric_resid_grad(self.train_tokens, metric_fn)
        self.metric_value = metric_value
        self.resids_pre = resids_pre
        self.resids_mid = resids_mid
        self.qk = qk
        self.grad_pre = grad_pre
        self.grad_mid = grad_mid

    def run_node_ies(self):
        self.ie_attn = {}
        self.sae_grads_attn = {}
        self.ie_resid = {}
        self.sae_grads_resid = {}
        self.ie_transcoder = {}
        self.sae_grads_transcoder = {}

        self.ie_error_attn = {}
        self.ie_error_resid = {}
        self.ie_error_transcoder = {}

        self.sae_error_attn = {}
        self.sae_error_resid = {}
        self.sae_error_transcoder = {}

        layers = self.layers
        for l in tqdm(layers):
            r_pre, r_mid, g_mid = self.resids_pre[l], self.resids_mid[l], self.grad_mid[l]
            sae = self.get_sae(layer=l, label="attn_out")
            indirect_effects, indirect_effects_error, sae_grad, error = sfc_simple(g_mid, r_mid - r_pre, r_mid - r_pre, sae)
            # display((indirect_effects > 0).sum(-1))
            self.ie_attn[l] = indirect_effects
            self.ie_error_attn[l] = indirect_effects_error
            self.sae_grads_attn[l] = sae_grad
            self.sae_error_attn[l] = error

        # for layer, (r_pre, g_pre) in enumerate(zip(resids_pre, grad_pre)):
        for l in tqdm(layers):
            r_pre, g_pre = self.resids_pre[l], self.grad_pre[l]
            sae = self.get_sae(layer=l)
            indirect_effects, indirect_effects_error, sae_grad, error = sfc_simple(g_pre, r_pre, r_pre, sae)
            # display((indirect_effects != 0).sum(-1))
            self.ie_resid[l] = indirect_effects
            self.ie_error_resid[l] = indirect_effects_error
            self.sae_grads_resid[l] = sae_grad
            self.sae_error_resid[l] = error

        for l in tqdm(layers[:-1]):
            r_mid, r_pre, g_pre = self.resids_mid[l], self.resids_pre[l + 1], self.grad_pre[l + 1]
            sae = self.get_sae(layer=l, label="transcoder")
            indirect_effects, indirect_effects_error, sae_grad, error = sfc_simple(g_pre, self.mlp_normalize(l, r_mid), r_pre - r_mid, sae)
            # display((indirect_effects != 0).sum(-1))
            self.ie_transcoder[l] = indirect_effects
            self.ie_error_transcoder[l] = indirect_effects_error
            self.sae_grads_transcoder[l] = sae_grad
            self.sae_error_transcoder[l] = error

    @eqx.filter_jit
    def run_with_add(self, additions_pre, additions_mid, tokens, metric, batched=False):
        get_resids = self.llama.select().at_instances_of(LlamaBlock).apply_with_selected_index(lambda i, x:
            pz.nn.Sequential([
                pz.de.TellIntermediate.from_config(tag=f"resid_pre_{i}"),
                x
            ])
        )
        get_resids = get_resids.select().at_instances_of(LlamaBlock).apply_with_selected_index(lambda l, b: b.select().at_instances_of(pz.nn.Residual).apply_with_selected_index(lambda i, x: x if i == 0 else pz.nn.Sequential([
            pz.de.TellIntermediate.from_config(tag=f"resid_mid_{l}"),
            x,
        ])))


        get_resids = get_resids.select().at_instances_of(LlamaAttention).apply_with_selected_index(lambda i, x: x.select().at_instances_of(pz.nn.Softmax).apply(lambda b: pz.nn.Sequential([
            b,
            pz.de.TellIntermediate.from_config(tag=f"attn_{i}"),
        ])))

        get_resids = pz.de.CollectingSideOutputs.handling(get_resids, tag_predicate=lambda x: True)
        make_additions = get_resids.select().at_instances_of(LlamaBlock).apply_with_selected_index(lambda i, x:
            pz.nn.Sequential([
                ActivationAddition(pz.nx.wrap(additions_pre[i], *(("batch",) if batched else ()), "seq", "embedding"), "all"),
                x
            ])
        )
        make_additions = make_additions.select().at_instances_of(LlamaBlock).apply_with_selected_index(lambda l, b: b.select().at_instances_of(pz.nn.Residual).apply_with_selected_index(lambda i, x: x if i == 0 else pz.nn.Sequential([
            ActivationAddition(pz.nx.wrap(additions_mid[l], *(("batch",) if batched else ()), "seq", "embedding"), "all"),
            x,
        ])))
        tokens_wrapped = pz.nx.wrap(tokens, "batch", "seq")
        logits, resids = make_additions(self.llama.inputs.from_basic_segments(tokens_wrapped))
        return metric(logits.unwrap("batch", "seq", "vocabulary"), resids, tokens), (logits, resids[::3], resids[1::3], resids[2::3])


    def get_metric_resid_grad(self, tokens, metric_fn):
        additions = [jnp.zeros(tokens.shape + (self.llama.config.hidden_size,)) for _ in range(self.llama.config.num_layers)]
        batched = tokens.ndim > 1
        (metric, (logits, resids_pre, qk, resids_mid)), (grad_pre, grad_mid) = jax.value_and_grad(self.run_with_add, argnums=(0, 1), has_aux=True)(additions, additions, tokens, metric_fn, batched=batched)
        return (
            metric,
            [r.value.unwrap("batch", "seq", "embedding") for r in resids_pre],
            [r.value.unwrap("batch", "seq", "embedding") for r in resids_mid],
            [r.value.unwrap("batch", "kv_heads", "q_rep", "seq", "kv_seq") for r in qk],
            grad_pre,
            grad_mid
        )


    def mlp_normalize(self, layer, resid_mid):
        return self.mlp_rms[layer](pz.nx.wrap(resid_mid, "batch", "seq", "embedding")).unwrap("batch", "seq", "embedding")

    def transcoder_feature_to_mid(self, layer, feature_idx, mask):
        sae = self.get_sae(layer=layer, label="transcoder")
        resid = self.resids_mid[layer]

        def f(resid):
            resid = self.mlp_normalize(layer, resid)
            batch_token_feat = resids_to_weights(resid, sae)[:, :, feature_idx] * self.sae_grads_transcoder[layer][:, :, feature_idx]
            token_act = self.mask_average(batch_token_feat, mask)
            return token_act

        return jax.grad(f)(resid)

    def transcoder_error_to_mid(self, layer, mask):
        sae = self.get_sae(layer=layer, label="transcoder")
        resid_next = self.resids_pre[layer + 1]
        resid = self.resids_mid[layer]

        grad = self.grad_pre[layer + 1]

        def f(resid):
            _, _, recon = sae_encode_gated(sae, resid)
            err_by_grad = jnp.einsum("...f, ...f -> ...", (resid_next - recon), grad)
            return self.mask_average(err_by_grad, mask)

        return jax.grad(f)(resid)
    def attn_out_feature_to_pre(self, layer, feature_idx, mask):
        sae = self.get_sae(layer=layer, label="attn_out")

        resid = self.resids_pre[layer]

        subblock = self.llama.select().at_instances_of(LlamaBlock).pick_nth_selected(layer).at_instances_of(pz.nn.Residual).pick_nth_selected(0).get().delta

        si_selection = subblock.select().at_instances_of(pz.de.HandledSideInputRef)
        keys = sorted(set([ref.tag for ref in si_selection.get_sequence()]))
        replaced = si_selection.apply(lambda ref: pz.de.SideInputRequest(tag=ref.tag))
        subblock = pz.de.WithSideInputsFromInputTuple.handling(replaced, keys)

        side_inputs = {
            'positions': self.llama_inputs.positions,
            'attn_mask': self.llama_inputs.attention_mask
        }

        def f(resid):
            resid = pz.nx.wrap(resid, "batch", "seq", "embedding")
            attn_out = subblock((resid,) + tuple(side_inputs[tag] for tag in subblock.side_input_tags))

            attn_out = attn_out.unwrap("batch", "seq", "embedding") 

            batch_token_feat = resids_to_weights(attn_out, sae)[:, :, feature_idx] * self.sae_grads_attn[layer][:, :, feature_idx]
            token_act = self.mask_average(batch_token_feat, mask)
            return token_act

        return jax.grad(f)(resid)

    def attn_out_error_to_pre(self, layer, mask):
        sae = self.get_sae(layer=layer, label="attn_out")

        resid = self.resids_pre[layer]

        subblock = self.llama.select().at_instances_of(LlamaBlock).pick_nth_selected(layer).at_instances_of(pz.nn.Residual).pick_nth_selected(0).get().delta

        si_selection = subblock.select().at_instances_of(pz.de.HandledSideInputRef)
        keys = sorted(set([ref.tag for ref in si_selection.get_sequence()]))
        replaced = si_selection.apply(lambda ref: pz.de.SideInputRequest(tag=ref.tag))
        subblock = pz.de.WithSideInputsFromInputTuple.handling(replaced, keys)

        side_inputs = {
            'positions': self.llama_inputs.positions,
            'attn_mask': self.llama_inputs.attention_mask
        }

        def f(resid):
            resid = pz.nx.wrap(resid, "batch", "seq", "embedding")
            attn_out = subblock((resid,) + tuple(side_inputs[tag] for tag in subblock.side_input_tags))

            attn_out = attn_out.unwrap("batch", "seq", "embedding") 

            _, _, recon = sae_encode_gated(sae, attn_out)
            batch_token_feat = jnp.einsum("...f, ...f -> ...", attn_out - recon, self.grad_mid[layer])
            token_act = self.mask_average(batch_token_feat, mask)
            return token_act

        return jax.grad(f)(resid)
    # float(jnp.linalg.norm(attn_out_error_to_pre(6, "arrow")))

    def pre_feature_to_pre(self, layer, feature_idx, mask):
        sae = self.get_sae(layer=layer)
        resid = self.resids_pre[layer]

        def f(resid):
            batch_token_feat = resids_to_weights(resid, sae)[:, :, feature_idx] * self.sae_grads_resid[layer][:, :, feature_idx]
            token_act = self.mask_average(batch_token_feat, mask)
            return token_act

        return jax.grad(f)(resid)

    def pre_error_to_pre(self, layer, mask):
        sae = self.get_sae(layer=layer)
        resid = self.resids_pre[layer]

        def f(resid):
            _, _, recon = sae_encode_gated(sae, resid)
            batch_token_error = jnp.einsum("...f, ...f -> ...", (resid - recon), self.grad_pre[layer])
            token_grad = self.mask_average(batch_token_error, mask)
            return token_grad

        return jax.grad(f)(resid)

    def ie_pre_to_transcoder_features(self, layer, grad, mask):
        sae = self.get_sae(layer=layer, label="transcoder")
        resid_mid = self.resids_mid[layer]
        resid_mid = self.mlp_normalize(layer, resid_mid)
        ie = sfc_simple(grad, resid_mid, resid_mid, sae)[0]
        ie = self.mask_average(ie, mask)

        return ie

    def ie_pre_to_transcoder_error(self, layer, grad, mask):
        sae = self.get_sae(layer=layer, label="transcoder")
        resid_next = self.resids_pre[layer + 1]
        resid_mid = self.resids_mid[layer]
        ie = sfc_simple(grad, self.mlp_normalize(layer, resid_mid), resid_next - resid_mid, sae)[1]
        ie = self.mask_average(ie, mask)

        return ie

    def ie_mid_to_attn_features(self, layer, grad, mask):
        sae = self.get_sae(layer=layer, label="attn_out")
        resid_mid = self.resids_mid[layer]
        resid_pre = self.resids_pre[layer]

        ie = sfc_simple(grad, resid_mid - resid_pre, resid_mid - resid_pre, sae)[0]
        ie = self.mask_average(ie, mask)
        return ie

    def ie_mid_to_attn_error(self, layer, grad, mask):
        sae = self.get_sae(layer=layer, label="attn_out")
        resid_mid = self.resids_mid[layer]
        resid_pre = self.resids_pre[layer]

        ie = sfc_simple(grad, resid_mid - resid_pre, resid_mid - resid_pre, sae)[1]
        ie = self.mask_average(ie, mask)
        return ie

    def ie_pre_to_pre_features(self, layer, grad, mask):
        sae = self.get_sae(layer=layer)
        resid = self.resids_pre[layer]
        ie = sfc_simple(grad, resid, resid, sae)[0]
        ie = self.mask_average(ie, mask)
        return ie

    def ie_pre_to_pre_error(self, layer, grad, mask):
        sae = self.get_sae(layer=layer)
        resid = self.resids_pre[layer]
        ie = sfc_simple(grad, resid, resid, sae)[1]
        ie = self.mask_average(ie, mask)
        return ie
    # float((ie_pre_to_pre_features(6, grad_pre[6], "arrow") - mask_average(ie_error_resid[6], "arrow")).sum())

    def mask_average(self, vector, mask):
        if isinstance(mask, jax.Array):
            mask = jax.lax.select_n(mask, *self.masks.values())
        else:
            mask = self.masks[mask]
        while mask.ndim < vector.ndim:
            mask = mask[..., None]

        return ((mask * vector).sum(1) / mask.sum(1)).mean(0)

    def grad_through_transcoder(self, layer, grad):
        sae = self.get_sae(layer, label="transcoder")
        resid_mid = self.resids_mid[layer]

        def f(resid_mid):
            resid_mid = self.mlp_normalize(layer, resid_mid)
            # we ignore error nodes
            weights = resids_to_weights(resid_mid, sae)
            recon = weights_to_resid(weights, sae)

            return recon

        grad = jax.vjp(f, resid_mid)[1](grad,)[0]

        return grad
    def grad_through_attn(self, layer, grad):
        subblock = self.llama.select().at_instances_of(LlamaBlock).pick_nth_selected(layer).at_instances_of(pz.nn.Residual).pick_nth_selected(0).get().delta

        si_selection = subblock.select().at_instances_of(pz.de.HandledSideInputRef)
        keys = sorted(set([ref.tag for ref in si_selection.get_sequence()]))
        replaced = si_selection.apply(lambda ref: pz.de.SideInputRequest(tag=ref.tag))
        subblock = pz.de.WithSideInputsFromInputTuple.handling(replaced, keys)

        side_inputs = {
            'positions': self.llama_inputs.positions,
            'attn_mask': self.llama_inputs.attention_mask
        }

        def f(resid):
            resid_pre = pz.nx.wrap(resid, "batch", "seq", "embedding")
            attn_out = subblock((resid_pre,) + tuple(side_inputs[tag] for tag in subblock.side_input_tags))

            attn_out = attn_out.unwrap("batch", "seq", "embedding") 

            return attn_out.astype(resid.dtype)

        resid = self.resids_pre[layer]
        return jax.vjp(f, resid)[1](grad.astype(resid.dtype),)[0]

    @eqx.filter_jit
    def ablated_metric(self, llama_ablated):
        ablated_logits = llama_ablated(self.llama_inputs)
        return metric_fn(ablated_logits.unwrap("batch", "seq", "vocabulary"), None, self.train_tokens, use_softmax=True)

    def mask_ie(self, ie, threshold, topk=None):
        out_masks = {}
        total_nodes = 0
        for mask in self.masks:
            ie_averaged = self.mask_average(ie, mask)
            ie_averaged = jnp.abs(ie_averaged)
            if topk is not None:
                i, w = jnp.lax.top_k(ie_averaged, topk)
                out_masks[mask] = jnp.zeros_like(ie_averaged).astype(bool).at[i].set(1)
            else:
                out_masks[mask] = jnp.abs(ie_averaged) > threshold
            total_nodes += out_masks[mask].sum()
        return out_masks, total_nodes

    @eqx.filter_jit
    def ablate_nodes(self, threshold, ablate_resids=False, topk=None):
        saes = self.saes
        ie_resid = self.ie_resid
        ie_attn, ie_transcoder = self.ie_attn, self.ie_transcoder
        llama_ablated = self.llama
        n_nodes = {0: 0}
        for layer in self.layers:
            block_selection = llama_ablated.select().at_instances_of(LlamaBlock).pick_nth_selected(layer)

            def converter(block):
                n_nodes_resid, n_nodes_attn, n_nodes_mlp = 0, 0, 0

                if ablate_resids:
                    try:
                        resid = saes[(layer, "resid")]
                        mask_resid, n_nodes_resid = self.mask_ie(ie_resid[layer], threshold, topk)
                        block = block.select().at_instances_of(LlamaBlock).apply(lambda x: pz.nn.Sequential([AblatedModule.wrap(resid, mask_resid, self.masks), x]))
                    except KeyError:
                        pass
                try:
                    attn_out = saes[(layer, "attn_out")]
                    mask_attn_out, n_nodes_attn = self.mask_ie(ie_attn[layer], threshold, topk)
                    block = block.select().at_instances_of(LlamaAttention).apply(lambda x: pz.nn.Sequential([x, AblatedModule.wrap(attn_out, mask_attn_out, self.masks)]))
                except KeyError:
                    pass

                try:
                    transcoder = saes[(layer, "transcoder")]
                    mask_transcoder, n_nodes_mlp = self.mask_ie(ie_transcoder[layer], threshold, topk)
                    block = block.select().at_instances_of(LlamaMLP).apply(lambda x: AblatedModule.wrap(transcoder, mask_transcoder, self.masks, x))
                except KeyError:
                    pass
                n_nodes[0] += n_nodes_attn + n_nodes_mlp + n_nodes_resid
                return block

            llama_ablated = block_selection.apply(converter)
        return self.ablated_metric(llama_ablated), n_nodes[0]

    def run_ablated_metrics(self, thresholds, topks=None):
        n_nodes_counts = []
        ablated_metrics = []

        if topks is not None:
            for topk in topks:
                abl_met, n_nodes = self.ablate_nodes(0, ablate_resids=True, topk=topk)
                ablated_metrics.append(float(abl_met))
                n_nodes_counts.append(int(n_nodes))
        else:
            for threshold in tqdm(thresholds):
                abl_met, n_nodes = self.ablate_nodes(threshold, ablate_resids=True, topk=topk)
                ablated_metrics.append(float(abl_met))
                n_nodes_counts.append(int(n_nodes))

        return ablated_metrics, n_nodes_counts

    from tqdm import tqdm, trange

    @eqx.filter_jit
    def compute_feature_effects(
        self,
        feature_type,
        layer,
        feature_idx,
        mask,
        layer_window=1,
    ):
        match feature_type:
            case "r":
                resid_grad = self.pre_feature_to_pre(layer, feature_idx, mask)
            case "t":
                resid_grad = self.transcoder_feature_to_mid(layer, feature_idx, mask)
            case "a":
                resid_grad = self.attn_out_feature_to_pre(layer, feature_idx, mask)
            case "er":
                resid_grad = self.pre_error_to_pre(layer, mask)
            case "et":
                resid_grad = self.transcoder_error_to_mid(layer, mask)
            case "ea":
                resid_grad = self.attn_out_error_to_pre(layer, mask)
        feature_effects = {}
        for l in range(layer, max(5, layer - (1 if feature_type in ("r", "er") else 0) - layer_window), -1):
            if l < layer:
                for mask in self.masks:
                    feature_effects[("t", l, mask)] = self.ie_pre_to_transcoder_features(l, resid_grad, mask)
                    feature_effects[("et", l, mask)] = self.ie_pre_to_transcoder_error(l, resid_grad, mask)
            # # does not work # resid_grad = resid_grad - grad_through_mlp(layer, resid_grad)
            # resid_grad = resid_grad + grad_through_mlp(layer, resid_grad)
            if l < layer or feature_type in ("t", "et"):
                for mask in self.masks:
                    feature_effects[("a", l, mask)] = self.ie_mid_to_attn_features(l, resid_grad, mask)
                    feature_effects[("ea", l, mask)] = self.ie_mid_to_attn_error(l, resid_grad, mask)
            # # does not work # resid_grad = resid_grad - grad_through_attn(layer, resid_grad)
            # resid_grad = resid_grad + grad_through_attn(layer, resid_grad)
            if l < layer or feature_type in ("t", "et", "a", "ea"):
                for mask in self.masks:
                    feature_effects[("r", l, mask)] = self.ie_pre_to_pre_features(l, resid_grad, mask)
                    feature_effects[("er", l, mask)] = self.ie_pre_to_pre_error(l, resid_grad, mask)
        return feature_effects

    def compute_edges(
        self,
        feature_type,
        layer,
        feature_idx,
        mask,
        abs_effects = False,
        k = 32,
        layer_window=1,
    ):
        feature_effects = self.compute_feature_effects(feature_type, layer, feature_idx, mask, layer_window=layer_window)
        top_effects = []
        for key, features in feature_effects.items():
            if features.ndim == 0:
                top_effects.append((float(features), key, 0))
                continue
            effects, indices = jax.lax.top_k(features if not abs_effects else jnp.abs(features), k)
            for i, e in zip(indices.tolist(), effects.tolist()):
                top_effects.append((e, key, i))
        top_effects.sort(reverse=True)
        return top_effects[:k]