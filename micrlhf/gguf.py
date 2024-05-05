# based on https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
# inspired by https://github.com/99991/pygguf/blob/main/gguf.py
import os
import struct

import numpy as np

GGUF_DATA_TYPE = {
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



class GGUFReader(object):
    def __init__(self, filename: os.PathLike):
        self.gguf_metadata, self.gguf_tensors = read_gguf_info(filename)
        self.mmap = np.memmap(filename)

    @property
    def metadata(self):
        return self.gguf_metadata

    def keys(self):
        return self.gguf_tensors.keys()
    
    def __iter__(self):
        return iter(self.gguf_tensors)

    def __getitem__(self, key):
        assert key in self.gguf_tensors
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
            assert ggml_type in GGUF_TENSOR_TYPES
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
    assert value_type in GGUF_DATA_TYPE
    value_type = GGUF_DATA_TYPE[value_type]
    if value_type == "string":
        return read_gguf_string(f)
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
