# based on https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
# inspired by https://github.com/99991/pygguf/blob/main/gguf.py
import os
import struct
from typing import Iterable

import numpy as np

GGUF_DATA_TYPE = {
    2: "uint16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64"
}
GGUF_TENSOR_TYPES = {
    0: "FP32",
    1: "FP16",
    8: "Q8_0",
    12: "Q4_K",
    14: "Q6_K"
}
GGUF_BLOCK_SIZES = {
    "FP32": 4,
    "FP16": 2,
    "Q8_0": 2 + 32,
    "Q4_K": 2 + 2 + 12 + 256 // 2,
    "Q6_K": 256 // 2 + 256 // 4 + 256 // 16 + 2,
}
GGUF_BLOCK_STRIDES = {
    "FP32": 1,
    "FP16": 1,
    "Q8_0": 32,
    "Q4_K": 256,
    "Q6_K": 256,
}
GGUF_DATA_TYPE_INV = {v: k for k, v in GGUF_DATA_TYPE.items()}


def read_gguf(filename: os.PathLike | Iterable[os.PathLike]):
    if isinstance(filename, (list, tuple)):
        return GGUFMultiplexer([GGUFReader(f) for f in filename])
    else:
        return GGUFReader(filename)

class GGUFMultiplexer(object):
    def __init__(self, ggufs):
        self.ggufs = ggufs

    def replace_metadata_prefix(self, prefix, new_prefix):
        for gguf in self.ggufs:
            gguf.replace_metadata_prefix(prefix, new_prefix)

    @property
    def metadata(self):
        return {k: v for gguf in self.ggufs for k, v in gguf.metadata.items()}

    def keys(self):
        return {k for gguf in self.ggufs for k in gguf.keys()}

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, key):
        for gguf in self.ggufs:
            try:
                return gguf[key]
            except KeyError:
                pass
        raise KeyError(f"Key {key} not found")


class GGUFReader(object):
    def __init__(self, filename: os.PathLike):
        self.gguf_metadata, self.gguf_tensors = read_gguf_info(filename)
        self.mmap = np.memmap(filename)

    def replace_metadata_prefix(self, prefix, new_prefix):
        self.gguf_metadata = {k.replace(prefix, new_prefix): v for k, v in self.gguf_metadata.items()}

    @property
    def metadata(self):
        return self.gguf_metadata

    def keys(self):
        return self.gguf_tensors.keys()
    
    def __iter__(self):
        return iter(self.gguf_tensors)

    def __getitem__(self, key):
        if key not in self.gguf_tensors:
            raise KeyError(f"Key {key} not found")
        tensor = self.gguf_tensors[key]
        start, end = tensor["offset"], tensor["offset"] + tensor["size"]
        data = self.mmap[start:end]
        if tensor["ggml_type"] == "FP32":
            return "fp32", (np.frombuffer(data, dtype=np.float32)[..., None],), tensor["shape"]
        elif tensor["ggml_type"] == "FP16":
            return "fp16", (np.frombuffer(data, dtype=np.float16)[..., None],), tensor["shape"]
        elif tensor["ggml_type"] == "Q8_0":
            scales = np.frombuffer(data, dtype=np.float16).reshape(-1, 1 + GGUF_BLOCK_STRIDES["Q8_0"] // 2)[:, :1]
            qs = np.frombuffer(data, dtype=np.int8).reshape(-1, 2 + GGUF_BLOCK_STRIDES["Q8_0"])[:, 2:]
            return "q8_0", (scales, qs), tensor["shape"]
        elif tensor["ggml_type"] == "Q4_K":
            # https://github.com/99991/pygguf/blob/829886d0726c89c6f6c0d8c39b0d507ec1604077/gguf.py#L206
            data_f16 = np.frombuffer(data, dtype=np.float16).reshape(-1, GGUF_BLOCK_SIZES["Q4_K"] // 2)
            data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(-1, GGUF_BLOCK_SIZES["Q4_K"])
            
            scale_factors = data_f16[:, 0].reshape(-1, 1)
            scale_offsets = data_f16[:, 1].reshape(-1, 1)
            qs1 = data_u8[:, 4:16].reshape(-1, 12)
            qs2 = data_u8[:, 16:].reshape(-1, 128)

            return "q4_k", (scale_factors, scale_offsets, qs1, qs2), tensor["shape"]
        elif tensor["ggml_type"] == "Q6_K":
            # https://github.com/99991/pygguf/blob/829886d0726c89c6f6c0d8c39b0d507ec1604077/gguf.py#L288
            data_f16 = np.frombuffer(data, dtype=np.float16).reshape(-1, GGUF_BLOCK_SIZES["Q6_K"] // 2)
            data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(-1, GGUF_BLOCK_SIZES["Q6_K"])
            data_i8 = np.frombuffer(data, dtype=np.int8).reshape(-1, GGUF_BLOCK_SIZES["Q6_K"])
            
            scales = data_f16[:, -1].reshape(-1, 1)
            ql = data_u8[:, :128].reshape(-1, 128).astype(np.int16)
            qh = data_u8[:, 128:192].reshape(-1, 64).astype(np.int16)
            sc = data_i8[:, 192:208].reshape(-1, 16).astype(np.float32)
            return "q6_k", (scales, ql, qh, sc), tensor["shape"]
        else:
            raise NotImplementedError(f"GGML type {tensor['ggml_type']} not implemented (yet)")


def read_gguf_info(filename: os.PathLike):
    with open(filename, "rb") as gguf:
        assert gguf.read(8) == b"GGUF\x03\x00\x00\x00"
        
        tensor_count = struct.unpack("<Q", gguf.read(8))[0]
        metadata_kv_count = struct.unpack("<Q", gguf.read(8))[0]
        
        kvs = [read_gguf_kv(gguf) for _ in range(metadata_kv_count)]
        metadata = {k: v for k, v in kvs}
        
        tensors = {}
        for _ in range(tensor_count):
            name = read_gguf_string(gguf)
            shape_len = read_gguf_value(gguf, GGUF_DATA_TYPE_INV["int32"])
            shape = [read_gguf_value(gguf, GGUF_DATA_TYPE_INV["uint64"]) for _ in range(shape_len)]
            ggml_type = read_gguf_value(gguf, GGUF_DATA_TYPE_INV["uint32"])
            assert ggml_type in GGUF_TENSOR_TYPES, f"Unknown GGML type {ggml_type} for tensor {name}"
            ggml_type = GGUF_TENSOR_TYPES[ggml_type]
            offset = read_gguf_value(gguf, GGUF_DATA_TYPE_INV["uint64"])
            size = np.prod(shape) * GGUF_BLOCK_SIZES[ggml_type] // GGUF_BLOCK_STRIDES[ggml_type]
            tensors[name] = {
                "shape": shape,
                "ggml_type": ggml_type,
                "offset": offset,
                "size": size,
            }

        end_of_header = gguf.tell()
        alignment = metadata.get("general.alignment", 32)
        tensors = {k: {**v, "offset": end_of_header + v["offset"]} for k, v in tensors.items()}
        tensors = {k: {**v, "offset": v["offset"] + (alignment - v["offset"] % alignment) % alignment} for k, v in tensors.items()}

    return metadata, tensors


def read_gguf_kv(f):
    key = read_gguf_string(f)
    value_type = struct.unpack("<I", f.read(4))[0]
    return key, read_gguf_value(f, value_type)


def read_gguf_value(f, value_type):
    assert value_type in GGUF_DATA_TYPE, "GGUF value not found!"
    value_type = GGUF_DATA_TYPE[value_type]
    if value_type == "string":
        return read_gguf_string(f)
    elif value_type == "uint16":
        return struct.unpack("<H", f.read(2))[0]
    elif value_type == "uint32":
        return struct.unpack("<I", f.read(4))[0]
    elif value_type == "int32":
        return struct.unpack("<i", f.read(4))[0]
    elif value_type == "float32":
        return struct.unpack("<f", f.read(4))[0]
    elif value_type == "uint64":
        return struct.unpack("<Q", f.read(8))[0]
    elif value_type == "array":
        data_type, count = struct.unpack("<IQ", f.read(4+8))
        return [read_gguf_value(f, data_type) for _ in range(count)]
    elif value_type == "bool":
        # read one byte and check if it's either 0 or 1
        byte = struct.unpack("<?", f.read(1))[0]
        assert byte in (0, 1)
        return byte


def read_gguf_string(f):
    # read string length
    str_len = struct.unpack("<Q", f.read(8))[0]
    # read string
    return f.read(str_len).decode("utf-8")


if __name__ == "__main__":
    gguf = GGUFReader("models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf")
    for k in gguf:
        v = gguf[k]
