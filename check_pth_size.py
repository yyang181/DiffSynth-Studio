import torch
import sys
from PIL import Image
from collections import deque
import os
import tqdm

def get_size(obj, seen=None):
    """递归估算 Python 对象真实占用内存（字节）"""
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    # Tensor
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()

    # PIL Image
    if isinstance(obj, Image.Image):
        mode_to_bpp = {
            "1": 1 / 8,  # 1 bit per pixel
            "L": 1, "P": 1,  # 8-bit pixels
            "RGB": 3, "YCbCr": 3,
            "RGBA": 4, "CMYK": 4, "I": 4, "F": 4
        }
        bpp = mode_to_bpp.get(obj.mode, 3)  # 默认当作 RGB
        width, height = obj.size
        return int(width * height * bpp)

    # dict
    if isinstance(obj, dict):
        return sum(get_size(k, seen) + get_size(v, seen) for k, v in obj.items())

    # list, tuple, set
    if isinstance(obj, (list, tuple, set, deque)):
        return sum(get_size(i, seen) for i in obj)

    # fallback
    return sys.getsizeof(obj)

def flatten_dict(d, parent_key=''):
    """展开嵌套字典以获取每个键值路径"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)

def check_pth(pth_path):
    # 加载 .pth 文件
    data = torch.load(pth_path, map_location='cpu')
    flat_data = flatten_dict(data)

    # 获取每项大小
    sizes = {k: get_size(v) for k, v in flat_data.items()}
    sorted_sizes = sorted(sizes.items(), key=lambda x: x[1], reverse=True)

    print("Top 10 largest entries (estimated memory):")
    for name, size in sorted_sizes[:10]:
        print(f"{name:<60} {size / 1024 / 1024:.2f} MB")

    total = sum(sizes.values())
    print(f"\nEstimated total size: {total / 1024 / 1024:.2f} MB")

# 工具：支持删除嵌套 key，如 'a.b.c'
def delete_key(d, key_path):
    parts = key_path.split('.')
    for p in parts[:-1]:
        if isinstance(d, dict) and p in d:
            d = d[p]
        else:
            return  # 不存在就跳过
    if isinstance(d, dict) and parts[-1] in d:
        print(f"Deleting key: {key_path}")
        del d[parts[-1]]

def clean_pth(pth_path, keys_to_remove, cleaned_pth_path):
    # 加载 .pth 文件
    data = torch.load(pth_path, map_location='cpu')

    # 删除指定的键
    for key in keys_to_remove:
        delete_key(data, key)

    torch.save(data, cleaned_pth_path)
    print(f"Cleaned .pth saved to: {cleaned_pth_path}, keys before cleaning: {len(data.keys())}, after cleaning: {len(data.keys())}")

if __name__ == "__main__":
    pth_dir = '/opt/data/private/yyx/data/OpenVidHD/train_pth'
    cleaned_pth_dir = '/opt/data/private/yyx/data/OpenVidHD/train_pth_cleaned'
    os.makedirs(cleaned_pth_dir, exist_ok=True)

    # 要删除的键名列表（支持嵌套路径）
    keys_to_remove = [
        'input_video',                      # 举例：顶层 key
        'vace_video',              # 举例：嵌套 key
    ]

    pth_files = sorted([f for f in os.listdir(pth_dir) if f.endswith('.pth')])
    for pth_file in tqdm.tqdm(pth_files, desc="Processing .pth files"):
        pth_path = os.path.join(pth_dir, pth_file)
        # print(f"Checking {pth_path}...")
        # check_pth(pth_path)

        cleaned_pth_path = os.path.join(cleaned_pth_dir, pth_file)
        clean_pth(pth_path, keys_to_remove, cleaned_pth_path)
        print(f"Cleaned {pth_file} and saved to {cleaned_pth_path}\n")

    print("All .pth files processed and cleaned.")
