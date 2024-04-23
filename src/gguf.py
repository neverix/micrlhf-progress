# based on https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
# loosely inspired by https://github.com/99991/pygguf/blob/main/gguf.py
import mmap
import struct


GGUF_DATA_TYPE = {
    4: "uint32",
    5: "int32",
    6: "float32",
    8: "string",
    9: "array",
    10: "uint64"
}
GGUF_DATA_TYPE_INV = {v: k for k, v in GGUF_DATA_TYPE.items()}


def read_gguf(filename: str):
    # read header and metadata
    with open(filename, "rb") as gguf:
        # handling version 3 only
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
            offset = read_gguf_value(gguf, GGUF_DATA_TYPE_INV["uint64"])
            tensors[name] = {
                "shape": shape,
                "ggml_type": ggml_type,
                "offset": offset
            }
        print(tensors.keys())


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


def read_gguf_string(f):
    # read string length
    str_len = struct.unpack("<Q", f.read(8))[0]
    # read string
    return f.read(str_len).decode("utf-8")


if __name__ == "__main__":
    read_gguf("models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf")
