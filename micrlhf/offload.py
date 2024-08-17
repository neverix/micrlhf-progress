import ctypes
import gc
import queue
import threading
import time
from queue import Queue
from threading import Thread

import jax
import jax.numpy as jnp
import jax_smi
import numpy as np
from ml_dtypes import bfloat16
from tqdm.auto import tqdm, trange

import splatmul
from splatmul import matmat, splatmat, splatsplat


def sparse_matmul(weights, indices, encoder_weight, decoder_weight, input_embeds, target_embeds):
    with jax.default_device(cpu):
        weights = np.asarray(weights).view(np.uint16)
        indices = np.asarray(indices).astype(np.uint32)
        decoder_weight = np.asarray(decoder_weight).view(np.uint16)
        encoder_weight = np.asarray(encoder_weight).view(np.uint16)
        result = splatmul.splatmul(weights, indices, decoder_weight)
        gc.collect()
        grads = None, None
        input_embeds = np.asarray(input_embeds)
        target_embeds = np.asarray(target_embeds)
        grads = splatmul.matsplat(input_embeds, target_embeds, encoder_weight, decoder_weight, weights, indices, result)
        del result, input_embeds, target_embeds, encoder_weight, decoder_weight, weights, indices
        gc.collect()
        return grads


def get_writeable_bit():
    max_arr_size = 128
    test_arr = np.array(1, dtype=np.uint16)
    ptr = id(test_arr)
    str_a = ctypes.string_at(ptr, max_arr_size)
    test_arr.setflags(write=0)
    str_b = ctypes.string_at(ptr, max_arr_size)
    writeable_bit = None
    for i in range(max_arr_size):
        if str_a[i] != str_b[i]:
            byte_a, byte_b = bin(str_a[i])[2:].zfill(8), bin(str_b[i])[2:].zfill(8)
            for j in range(8):
                if byte_a[j] != byte_b[j]:
                    writeable_bit = i * 8 + j
    writeable_byte = writeable_bit // 8, writeable_bit % 8
    return writeable_byte


def transmogrify(x):
    byte, bit = get_writeable_bit()
    ptr = id(x)
    current = ctypes.string_at(ptr + byte, 1)
    target = current[0] | (1 << (7-bit))
    ctypes.memset(ptr + byte, target, 1)
    return x

if __name__ == "__main__":
    input_size = 2**18
    hidden_size = 2**14
    w_chunk = 2**12
    w_size = 2**20
    k = 64
    encoder_scale = 300
    n_dp, n_mp = 4, 1

    lr = 1e-4
    # beta1 = 0.9
    beta1 = 0.0
    beta2 = 0.98
    input_size = input_size * n_dp

    jax_smi.initialise_tracking()
    devices = np.asarray(jax.local_devices()).reshape(n_dp, n_mp)
    mesh = jax.sharding.Mesh(devices, ("dp", "mp"))

    cpu = jax.devices("cpu")[0]
    to_cpu = lambda x: jax.device_put(x, cpu)
    with jax.default_device(cpu):
        weight_enc, weight_dec = map(lambda x: jnp.asarray(x).view(jnp.bfloat16), matmat(w_size, hidden_size))
    adam_state_encoder = splatsplat(w_size, hidden_size,
                            lr, 0.9, 0.999, 1e-8, 16)
    adam_state_decoder = splatsplat(w_size, hidden_size,
                            lr, 0.9, 0.999, 1e-8, 16)


    key_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp", None))
    weight_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "mp"))
    topk_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp", "mp"))

    @partial(jax.jit, in_shardings=(key_sharding,), out_shardings=(data_sharding, key_sharding), donate_argnums=(0,))
    def gen_inputs(input_key):
        input_key, subkey = input_key + 1, input_key
        inputs = jax.lax.broadcasted_iota(jnp.int4, (input_size, hidden_size), 0).astype(jnp.int8) + jax.lax.broadcasted_iota(jnp.int4, (input_size, hidden_size), 1).astype(jnp.int8) + subkey
        return inputs, input_key

    @partial(jax.jit, in_shardings=(data_sharding, weight_sharding, topk_sharding, topk_sharding, None), out_shardings=(topk_sharding, topk_sharding), donate_argnums=(2, 3))
    def matmul_topk_jit(inputs, weight_chunk, old_weights, old_indices, offset):
        output_chunk = inputs @ weight_chunk
        weights, indices = jax.lax.approx_max_k(output_chunk.astype(jnp.bfloat16), k=k, recall_target=0.8)
        weights = weights * (encoder_scale / 127.5)
        indices += offset
        
        if old_weights is None or old_indices is None:
            return weights, indices
        else:
            new_full_weights = jnp.concatenate((weights, old_weights), axis=1)
            new_full_indices = jnp.concatenate((indices, old_indices), axis=1)

            _, over_indices = jax.lax.top_k(new_full_weights, k=k)
            return jnp.take_along_axis(new_full_weights, over_indices, axis=1), jnp.take_along_axis(new_full_indices, over_indices, axis=1)

    def send_weights(weights):
        return jax.device_put(weights, weight_sharding)

    def sparse_matmul_async(weights, indices, past_embeds=None):
        out_queue = Queue()
        def worker():
            out_queue.put(sparse_matmul(weights, indices, weight_enc, weight_dec, past_embeds, past_embeds))
        thread = Thread(target=worker)
        thread.start()
        return out_queue


    def matmul_trial(inputs):
        with jax.default_device(cpu):
            chunk = weight_enc[:w_chunk * n_mp].T
        current_chunk = send_weights(chunk)
        weights, indices = None, None
        bar = trange(0, w_size, w_chunk * n_mp, postfix=f"Encoder forward pass")
        offset = 0
        for chunk_start in bar:
            with jax.default_device(cpu):
                weight_enc[chunk_start:chunk_start + w_chunk * n_mp].T
            next_chunk = send_weights(chunk)
            weights, indices = matmul_topk_jit(inputs, current_chunk, weights, indices, offset)
            offset += w_chunk * n_mp
            current_chunk = next_chunk
            gc.collect()
        return weights, indices


    def trial(save_encodings=False):
        global saved_encodings
        gc.collect()
        input_key = 0
        encodings, inputs, decoder_outputs, grads = None, None, None, None
        for _ in trange(100, postfix="Measuring forward speed..."):
            gc.collect()
            if encodings is not None:
                past_inputs = jax.device_put(inputs, cpu)
                past_embeds = np.asarray(past_inputs)
                decoder_outputs = sparse_matmul_async(encodings[0], encodings[1], past_embeds=past_embeds)
                del past_inputs, inputs, encodings, past_embeds
                gc.collect()
            inputs, input_key = gen_inputs_jit(input_key)
            weights, indices = matmul_trial(inputs)
            if decoder_outputs is not None:
                print("Computing decoder/grads...")
                decoder_start = time.time()
                grads = decoder_outputs.get()
                decoder_end = time.time()
                print("Decoder/grad time:", decoder_end - decoder_start)
                gc.collect()
            if grads is not None:
                print("Doing update...")
                start_update = time.time()
                with jax.default_device(cpu):
                    weight_dec_np = transmogrify(np.asarray(weight_dec).view(np.uint16))
                    splatmat(adam_state_decoder, grads, weight_dec_np, is_decoder=False)
                    weight_enc_np = transmogrify(np.asarray(weight_enc).view(np.uint16))
                    splatmat(adam_state_encoder, grads, weight_enc_np, is_decoder=True)
                    del grads, weight_enc_np, weight_dec_np
                    gc.collect()
                print("Update time:", time.time() - start_update)
            print("Waiting for encoder...")
            before_compute = time.time()
            weights, indices = weights.block_until_ready(), indices.block_until_ready()
            after_compute = time.time()
            print("Encoder time:", (after_compute - before_compute))
            print("Sending encodings to CPU...")
            start_send = time.time()
            weights = to_cpu(weights)
            indices = to_cpu(indices)
            encodings = weights, indices
            del weights, indices
            gc.collect()
            print("Sent encodings to CPU. Time:", time.time() - start_send)
            if save_encodings:
                saved_encodings = encodings
                break

    trial()