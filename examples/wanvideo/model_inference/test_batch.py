import os
import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import argparse
from safetensors.torch import load_file, save_file
from modelscope import snapshot_download, dataset_snapshot_download
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_inference(checkpoint_path):
    # 与你原代码一致的保存路径逻辑
    save_folder = os.path.join(
        'output',
        checkpoint_path.split(os.sep)[-2],
        os.path.basename(checkpoint_path).split('.')[0]
    )
    os.makedirs(save_folder, exist_ok=True)

    output_path = os.path.join(save_folder, "structure.mp4")
    if os.path.exists(output_path):
        print(f"[✓] Skipping {checkpoint_path}, output already exists at {output_path}")
        return

    print(f"[→] Running inference for {checkpoint_path} ...")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=['models/iic/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors', checkpoint_path], offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )
    pipe.enable_vram_management()

    control_video = VideoData("data/example_video_dataset/video1_softedge.mp4", height=480, width=832)
    num_frames = len(control_video)
    video = pipe(
        prompt="from sunset to night, a small town, light, house, river",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        vace_video=control_video,
        vace_reference_image=Image.open("data/example_video_dataset/reference_image.png").resize((832, 480)),
        seed=1, tiled=True, num_frames=num_frames
    )
    save_video(video, output_path, fps=15, quality=5)
    print(f"[✓] Saved result to {output_path}")

def extract_epoch_number(filename):
    match = re.search(r'epoch-(\d+)', filename)
    return int(match.group(1)) if match else -1
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffSynth auto checkpoint inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path (file or folder)")
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if os.path.isfile(ckpt_path):
        run_inference(ckpt_path)
    elif os.path.isdir(ckpt_path):
        # ckpt_files = [os.path.join(ckpt_path, f) for f in os.listdir(ckpt_path) if f.endswith(".safetensors")]
        ckpt_files = [
            os.path.join(ckpt_path, f)
            for f in os.listdir(ckpt_path)
            if f.endswith(".safetensors") and "epoch-" in f
        ]
        ckpt_files.sort(key=lambda x: extract_epoch_number(os.path.basename(x)), reverse=True)

        for ckpt in ckpt_files:
            # print(f"Processing checkpoint: {ckpt}")
            # continue 
            
            print(f"Inferencing with checkpoint: {ckpt}")
            run_inference(ckpt)
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")
