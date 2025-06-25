import os
import torch
from PIL import Image
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new_v1_vacefull import ModelConfig
from diffsynth.pipelines.wan_video_new_v1_vacefull import WanVideoPipeline_v1_vacefull as WanVideoPipeline
import argparse
from safetensors.torch import load_file, save_file
from modelscope import snapshot_download, dataset_snapshot_download
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_inference(checkpoint_path):
    test_list_path = [
        ["/opt/data/private/yyx/data/20241210_RVSR_test/flow_propagation/ref/1984/frame0054.png",
         "/opt/data/private/yyx/data/20241210_RVSR_test/flow_propagation/input_video/1984.mp4",
         "The video begins with a close-up shot of a person holding a red flag on a track. The individual is dressed in formal attire, suggesting a ceremonial or official role. The background is a vibrant orange, likely part of a stadium or arena setting, which is partially visible. The name \"LIBIN WANG\" appears at the bottom left corner of the screen, accompanied by a logo featuring a stylized figure running"],

        ["/opt/data/private/yyx/data/OpenVidHD_onlyonemp4/images/0zT5Ux__Pbk_16_0to101/frame0000.png",
         "/opt/data/private/yyx/data/OpenVidHD_onlyonemp4/lr_videos/lr_0zT5Ux__Pbk_16_0to101.mp4",
         "The video features a bald man with a beard, wearing a black t-shirt, sitting in the driver's seat of a car with the sunroof open. The car is moving, as indicated by the blurred background of trees and sky. The man appears to be speaking, possibly giving a review or commentary about the car. The style of the video is casual and informal, with a focus on the man's reactions and expressions. The lighting is bright, suggesting it is daytime."],
    ]

    # 通用保存根路径
    save_root = os.path.join(
        'output',
        checkpoint_path.split(os.sep)[-2],
        os.path.basename(checkpoint_path).split('.')[0]
    )
    os.makedirs(save_root, exist_ok=True)

    print(f"[→] Initializing pipeline for {checkpoint_path} ...")

    # add try except block to handle potential errors
    try:
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(path=[checkpoint_path
                ], offload_device="cpu"),
                ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B",
                            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                            offload_device="cpu"),
                ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B",
                            origin_file_pattern="Wan2.1_VAE.pth",
                            offload_device="cpu"),
            ],
        )

        ### add vacefull parameters
        pipe.dit.enable_vacefull_condition()
        print(f"🔍 Loading weights from: {checkpoint_path}")
        dit_state_dict = load_state_dict(checkpoint_path)
        vacefull_state_dict = {}
        for key in dit_state_dict.keys():
            if 'vacefull_patch_embedding.' in key:
                vacefull_state_dict[key.split("vacefull_patch_embedding.")[1]] = dit_state_dict[key]
        pipe.dit.vacefull_patch_embedding.load_state_dict(vacefull_state_dict, strict=True)


        pipe.enable_vram_management()

        ### add torch compile
        pipe.vae = torch.compile(pipe.vae, mode="default")  # 编译 VAE 模块
        pipe.dit = torch.compile(pipe.dit, mode="default")  # 编译 DIT 模块
        pipe.text_encoder = torch.compile(pipe.text_encoder, mode="default")  # 编译文本编码器

    except Exception as e:
        print(f"[✗] Failed to initialize pipeline with {checkpoint_path}: {e}")
        return

    for i, (ref_img_path, driving_video_path, prompt) in enumerate(test_list_path):
        video_name = os.path.splitext(os.path.basename(driving_video_path))[0]
        output_path = os.path.join(save_root, f"{video_name}.mp4")

        if os.path.exists(output_path):
            print(f"[✓] Skipping sample {i} ({video_name}), output already exists.")
            continue

        print(f"[→] Running sample {i}: {video_name}")

        control_video = VideoData(driving_video_path, height=480, width=832)

        # Ensure the video has at most 49 frames
        control_video = [control_video[i] for i in range(49)] if len(control_video) > 49 else control_video
        num_frames = len(control_video)
        # print(f"Number of frames in the video: {num_frames}");assert 0

        video = pipe(
            prompt=prompt,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            vace_video=control_video,
            vace_reference_image=Image.open(ref_img_path).resize((832, 480)),
            seed=1,
            tiled=True,
            num_frames=num_frames
        )

        save_video(video, output_path, fps=15, quality=5)
        print(f"[✓] Saved result to {output_path}")


# def run_inference(checkpoint_path):
#     test_list_path= [
#         # Format: [reference image, driving pose sequence, text prompt]
#         ["/opt/data/private/yyx/data/20241210_RVSR_test/flow_propagation/ref/1984/frame0054.png", "/opt/data/private/yyx/data/20241210_RVSR_test/flow_propagation/input_video/1984.mp4", "The video begins with a close-up shot of a person holding a red flag on a track. The individual is dressed in formal attire, suggesting a ceremonial or official role. The background is a vibrant orange, likely part of a stadium or arena setting, which is partially visible. The name \"LIBIN WANG\" appears at the bottom left corner of the screen, accompanied by a logo featuring a stylized figure running"],
#         ["/opt/data/private/yyx/data/OpenVidHD_onlyonemp4/images/0zT5Ux__Pbk_16_0to101/frame0000.png", "/opt/data/private/yyx/data/OpenVidHD_onlyonemp4/lr_videos/lr_0zT5Ux__Pbk_16_0to101.mp4", "The video features a bald man with a beard, wearing a black t-shirt, sitting in the driver's seat of a car with the sunroof open. The car is moving, as indicated by the blurred background of trees and sky. The man appears to be speaking, possibly giving a review or commentary about the car. The style of the video is casual and informal, with a focus on the man's reactions and expressions. The lighting is bright, suggesting it is daytime."],
#     ]


#     # 与你原代码一致的保存路径逻辑
#     save_folder = os.path.join(
#         'output',
#         checkpoint_path.split(os.sep)[-2],
#         os.path.basename(checkpoint_path).split('.')[0]
#     )
#     os.makedirs(save_folder, exist_ok=True)

#     output_path = os.path.join(save_folder, "structure.mp4")
#     if os.path.exists(output_path):
#         print(f"[✓] Skipping {checkpoint_path}, output already exists at {output_path}")
#         return

#     print(f"[→] Running inference for {checkpoint_path} ...")

#     pipe = WanVideoPipeline.from_pretrained(
#         torch_dtype=torch.bfloat16,
#         device="cuda",
#         model_configs=[
#             ModelConfig(path=['models/Wan-AI/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors', checkpoint_path], offload_device="cpu"),
#             ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
#             ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
#         ],
#     )
#     pipe.enable_vram_management()

#     control_video = VideoData("data/example_video_dataset/video1_softedge.mp4", height=480, width=832)
#     num_frames = len(control_video)
#     video = pipe(
#         prompt="from sunset to night, a small town, light, house, river",
#         negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#         vace_video=control_video,
#         vace_reference_image=Image.open("data/example_video_dataset/reference_image.png").resize((832, 480)),
#         seed=1, tiled=True, num_frames=num_frames
#     )
#     save_video(video, output_path, fps=15, quality=5)
#     print(f"[✓] Saved result to {output_path}")

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
            if f.endswith(".safetensors") and "epoch-" in f and not f.startswith(".")
        ]
        ckpt_files.sort(key=lambda x: extract_epoch_number(os.path.basename(x)), reverse=True)

        for ckpt in ckpt_files:
            # print(f"Processing checkpoint: {ckpt}")
            # continue 
            
            print(f"Inferencing with checkpoint: {ckpt}")
            run_inference(ckpt)
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")
