import os
import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import argparse
from safetensors.torch import load_file, save_file
from modelscope import snapshot_download, dataset_snapshot_download

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# prompt = 'A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.'
# save_name = 'woman_walks_in_the_park'
prompt_txt_path = '/opt/data/private/yyx/code/Self-Forcing/prompts/MovieGenVideoBench_extended.txt'
# read prompt from txt
with open(prompt_txt_path, 'r') as f:
    prompts = f.readlines()
    prompts = [line.strip() for line in prompts if line.strip()]
    # avoid save_name too long 
    save_names = [line.split(',')[0].strip()[:20] for line in prompts if line.strip()]

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        # ModelConfig(model_id=model_id_with_origin_paths.split(":")[0], origin_file_pattern=model_id_with_origin_paths.split(":")[1], offload_device="cpu"),
        # ModelConfig(path=['models/iic/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors', args.checkpoint], offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()

### add torch compile
pipe.vae = torch.compile(pipe.vae, mode="default")  # 编译 VAE 模块
pipe.dit = torch.compile(pipe.dit, mode="default")  # 编译 DIT 模块
pipe.text_encoder = torch.compile(pipe.text_encoder, mode="default")  # 编译文本编码器


for i, (prompt, save_name) in enumerate(zip(prompts, save_names)):
    print(f"[→] Running inference for {save_name} ...")
    save_folder = os.path.join('output', 'Wan2.1-T2V-1.3B')
    os.makedirs(save_folder, exist_ok=True)
    
    output_path = os.path.join(save_folder, f"{save_name}.mp4")
    if os.path.exists(output_path):
        print(f"[✓] Skipping {save_name}, output already exists at {output_path}")
        continue

    # Text-to-video
    video = pipe(
        prompt=prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        seed=0, tiled=True,
    )
    save_video(video, output_path, fps=15, quality=5)

    # # Video-to-video
    # video = VideoData(os.path.join(save_folder, "video1.mp4"), height=480, width=832)
    # video = pipe(
    #     prompt="纪实摄影风格画面，一只活泼的小狗戴着黑色墨镜在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，戴着黑色墨镜，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
    #     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    #     input_video=video, denoising_strength=0.7,
    #     seed=1, tiled=True
    # )

    # # Depth video + Reference image -> Video
    # control_video = VideoData("data/examples/wan/depth_video.mp4", height=480, width=832)
    # num_frames = len(control_video)
    # video = pipe(
    #     prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
    #     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    #     vace_video=control_video,
    #     vace_reference_image=Image.open("data/examples/wan/cat_fightning.jpg").resize((832, 480)),
    #     seed=1, tiled=True, num_frames=num_frames
    # )
    # save_video(video, os.path.join(save_folder, "cat.mp4"), fps=15, quality=5)

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
    # save_video(video, os.path.join(save_folder, "structure.mp4"), fps=15, quality=5)
