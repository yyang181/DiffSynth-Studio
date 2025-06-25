import hashlib
from safetensors.torch import load_file
import argparse
import torch 
from safetensors import safe_open

def load_state_dict_from_safetensors(file_path, torch_dtype=None, device="cpu"):
    state_dict = {}
    with safe_open(file_path, framework="pt", device=device) as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict

def hash_state_dict_keys(state_dict, with_shape=True):
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    return hashlib.md5(keys_str).hexdigest()

def convert_state_dict_keys_to_single_str(state_dict, with_shape=True):
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor):
                if with_shape:
                    shape = "_".join(map(str, list(value.shape)))
                    keys.append(key + ":" + shape)
                keys.append(key)
            elif isinstance(value, dict):
                keys.append(key + "|" + convert_state_dict_keys_to_single_str(value, with_shape=with_shape))
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str

# def get_state_dict_keys_hash_with_shape_sha256(state_dict):
#     """
#     使用 SHA256 哈希生成 state_dict_keys_hash_with_shape。
#     格式：每个键及其 shape 组成 "key:shape" 字符串，并排序后拼接哈希。
#     """
#     entries = [f"{k}:{tuple(v.shape)}" for k, v in state_dict.items()]
#     entries_str = "\n".join(sorted(entries)).encode("utf-8")
#     # entries_str = "\n".join((entries)).encode("utf-8")
#     return hashlib.sha256(entries_str).hexdigest()

def main(path):
    print(f"🔍 Loading weights from: {path}")
    state_dict = load_state_dict_from_safetensors(path)
    hash_str = hash_state_dict_keys(state_dict, with_shape=True)
    print(f"\n✅ SHA256 state_dict_keys_hash_with_shape:\n{hash_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str, help="Path to .safetensors file")
    args = parser.parse_args()
    main(args.ckpt_path)
