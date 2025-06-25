import hashlib
from safetensors.torch import load_file
import argparse

def get_state_dict_keys_hash_with_shape_sha256(state_dict):
    """
    使用 SHA256 哈希生成 state_dict_keys_hash_with_shape。
    格式：每个键及其 shape 组成 "key:shape" 字符串，并排序后拼接哈希。
    """
    entries = [f"{k}:{tuple(v.shape)}" for k, v in state_dict.items()]
    entries_str = "\n".join(sorted(entries)).encode("utf-8")
    # entries_str = "\n".join((entries)).encode("utf-8")
    return hashlib.sha256(entries_str).hexdigest()

def main(path):
    print(f"🔍 Loading weights from: {path}")
    state_dict = load_file(path)
    hash_str = get_state_dict_keys_hash_with_shape_sha256(state_dict)
    print(f"\n✅ SHA256 state_dict_keys_hash_with_shape:\n{hash_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str, help="Path to .safetensors file")
    args = parser.parse_args()
    main(args.ckpt_path)
