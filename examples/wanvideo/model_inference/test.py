import os
import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import argparse
from safetensors.torch import load_file, save_file
from modelscope import snapshot_download, dataset_snapshot_download

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="models/iic/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors",
    help="Task. `data_process` or `train`.",
)
args = parser.parse_args()

model_id_with_origin_paths = args.checkpoint


save_folder = os.path.join('output', model_id_with_origin_paths.split(os.sep)[-2], model_id_with_origin_paths.split(os.sep)[-1].split('.')[0])
os.makedirs(save_folder, exist_ok=True)

# Download example video
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=["data/examples/wan/depth_video.mp4", "data/examples/wan/cat_fightning.jpg"]
)

# check if pretrained ckpt have same content as finetuned ckpt
if 0:
    ckpt1 = 'models/iic/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors'
    ckpt2 = model_id_with_origin_paths

    # 加载权重（不放到GPU）
    state_dict1 = load_file(ckpt1, device="cpu")
    state_dict2 = load_file(ckpt2, device="cpu")
    shape_mismatch = []
    key_value_mismatch = []
    key_value_match = []
    key_not_found = []

    # save keys in ckpt2 to txt 
    with open(os.path.join(save_folder, "keys_in_ckpt1.txt"), "w") as f:
        for key in state_dict1.keys():
            f.write(f"{key}\n")
    print(f"Total keys in ckpt1: {len(state_dict1.keys())}")

    with open(os.path.join(save_folder, "keys_in_ckpt2.txt"), "w") as f:
        for key in state_dict2.keys():
            f.write(f"{key}\n")
    print(f"Total keys in ckpt2: {len(state_dict2.keys())}")
    assert 0

    for key in state_dict1.keys():
        if key not in state_dict2:
            print(f"Key {key} not found in second state_dict")
            key_not_found.append(key)
        else:
            # Compare the tensors
            tensor1 = state_dict1[key]
            tensor2 = state_dict2[key]
            # if not key.startswith("block"):
            if 1:
                if tensor1.shape != tensor2.shape:
                    print(f"Shape mismatch for key {key}: {tensor1.shape} vs {tensor2.shape}")
                    shape_mismatch.append(key)
                else:
                    # 转换为相同的 dtype 再比较
                    t1 = tensor1.to(torch.float32)
                    t2 = tensor2.to(torch.float32)
                if not torch.allclose(t1, t2, atol=1e-6):
                    print(f"Tensor value mismatch for key {key}")
                    key_value_mismatch.append(key)
                else:
                    print(f"Key {key} is the same in both state_dicts")
                    key_value_match.append(key)
    print(f"Shape mismatches: {len(shape_mismatch)}")
    print(f"Key value mismatches: {len(key_value_mismatch)}")
    print(f"Key value matches: {len(key_value_match)}")
    print(f"Keys not found in second state_dict: {len(key_not_found)}")
    print(f"Total keys in first state_dict: {len(state_dict1.keys())}")
    print(f"Total keys in second state_dict: {len(state_dict2.keys())}")
    print(key_not_found)

    assert 0

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        # ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        # ModelConfig(model_id=model_id_with_origin_paths.split(":")[0], origin_file_pattern=model_id_with_origin_paths.split(":")[1], offload_device="cpu"),
        ModelConfig(path=['models/iic/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors', args.checkpoint], offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()

# # Text-to-video
# video = pipe(
#     prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
# )
# save_video(video, os.path.join(save_folder, "video1.mp4"), fps=15, quality=5)

# # Video-to-video
# video = VideoData(os.path.join(save_folder, "video1.mp4"), height=480, width=832)
# video = pipe(
#     prompt="纪实摄影风格画面，一只活泼的小狗戴着黑色墨镜在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，戴着黑色墨镜，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     input_video=video, denoising_strength=0.7,
#     seed=1, tiled=True
# )

# Depth video + Reference image -> Video
control_video = VideoData("data/examples/wan/depth_video.mp4", height=480, width=832)
num_frames = len(control_video)
video = pipe(
    prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    vace_video=control_video,
    vace_reference_image=Image.open("data/examples/wan/cat_fightning.jpg").resize((832, 480)),
    seed=1, tiled=True, num_frames=num_frames
)
save_video(video, os.path.join(save_folder, "example_video_dataset.mp4"), fps=15, quality=5)

# # reference to video 
# control_video = VideoData("data/example_video_dataset/video1_softedge.mp4", height=480, width=832)
# num_frames = len(control_video)
# video = pipe(
#     prompt="from sunset to night, a small town, light, house, river",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     vace_video=control_video,
#     vace_reference_image=Image.open("data/example_video_dataset/reference_image.png").resize((832, 480)),
#     seed=1, tiled=True, num_frames=num_frames
# )
# save_video(video, os.path.join(save_folder, "example_video_dataset2.mp4"), fps=15, quality=5)
