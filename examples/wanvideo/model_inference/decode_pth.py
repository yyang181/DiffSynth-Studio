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
import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def check_pth_keys(pth_dir):
    pth_files = sorted([f for f in os.listdir(pth_dir) if f.endswith('.pth')])
    pth_files.sort(key=lambda x: int(x.split(".")[0]))
    for pth_file in tqdm.tqdm(pth_files, desc="Checking .pth files"):
        pth_path = os.path.join(pth_dir, pth_file)
        data = torch.load(pth_path, map_location='cpu')
        print(f"Keys in {pth_file}: {list(data.keys())}")
        assert 0
        # Keys in 0.pth: ['height', 'width', 'num_frames', 'cfg_scale', 'tiled', 'rand_device', 'use_gradient_checkpointing', 'use_gradient_checkpointing_offload', 'cfg_merge', 'vace_scale', 'noise', 'latents', 'input_latents', 'vace_context', 'vace_video_latent', 'vace_reference_latent', 'prompt', 'context', 'negative_context']

def decode_pth_keys(pth_dir, decode_dir, pipe):
    pth_files = sorted([f for f in os.listdir(pth_dir) if f.endswith('.pth')])
    # pth_files.sort(key=lambda x: int(x.split(".")[0]))

    pipe.load_models_to_device(['vae'])

    for pth_file in tqdm.tqdm(pth_files, desc="Checking .pth files"):
        pth_path = os.path.join(pth_dir, pth_file)
        data = torch.load(pth_path, map_location='cpu')

        keys_to_ignonre = ['vace_context', 'context', 'negative_context'
        ]

        # check all keys' shape and save to txt file 
        os.makedirs(os.path.dirname(os.path.join(decode_dir, pth_file)), exist_ok=True)
        with open(os.path.join(decode_dir, pth_file + '.txt'), 'w') as f:
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    f.write(f"{key}: {data[key].shape, type(data[key])}\n") 
                elif isinstance(data[key], str):
                    f.write(f"{key}: {data[key], type(data[key])}\n")
                elif isinstance(data[key], int):
                    f.write(f"{key}: {str(data[key]), type(data[key])}\n")
                elif isinstance(data[key], float):
                    f.write(f"{key}: {str(data[key]), type(data[key])}\n")
                elif isinstance(data[key], bool):
                    f.write(f"{key}: {str(data[key]), type(data[key])}\n")
                elif isinstance(data[key], torch.device):
                    f.write(f"{key}: {str(data[key]), type(data[key])}\n")
            f.write(f"total lens of keys: {len(data)}\n")
        # assert 0


        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if key in keys_to_ignonre:
                    print(f"Skipping {key} in {pth_file}...")
                    continue

                data[key] = data[key].to(dtype=pipe.torch_dtype, device=pipe.device)
                print(f"************* Decoding {key} in {pth_file} *************")
                # continue

                if data[key].shape[2] == 14:
                    # print(key, data[key].shape);assert 0 # noise torch.Size([1, 16, 14, 60, 104])

                    # save image 
                    output_path_image = os.path.join(decode_dir, pth_file, f"{key}.png")
                    os.makedirs(os.path.dirname(output_path_image), exist_ok=True)
                    image = pipe.vae.decode(data[key][:,:,0,:,:].unsqueeze(2), device=pipe.device, tiled=True, tile_size=(30, 52), tile_stride=(15, 26))
                    image = pipe.vae_output_to_video(image)
                    image[0].save(output_path_image, format='PNG')
                
                    # save video 
                    real_video = data[key][:,:,1:,:,:]
                    # print(real_video.shape); assert 0 # torch.Size([1, 16, 13, 60, 104])
                    real_video = pipe.vae.decode(real_video, device=pipe.device, tiled=True, tile_size=(30, 52), tile_stride=(15, 26))
                    video = pipe.vae_output_to_video(real_video)

                    output_path = os.path.join(decode_dir, pth_file, f"{key}.mp4")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    save_video(video, output_path, fps=15, quality=5)

                elif data[key].shape[2] == 13:
                    video = pipe.vae.decode(data[key], device=pipe.device, tiled=True, tile_size=(30, 52), tile_stride=(15, 26))
                    video = pipe.vae_output_to_video(video)

                    output_path = os.path.join(decode_dir, pth_file, f"{key}.mp4")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    save_video(video, output_path, fps=15, quality=5)

                elif data[key].shape[2] == 1 and key == 'vace_reference_latent':
                    video = pipe.vae.decode(data[key], device=pipe.device, tiled=True, tile_size=(30, 52), tile_stride=(15, 26))
                    video = pipe.vae_output_to_video(video)

                    output_path = os.path.join(decode_dir, pth_file, f"{key}.png")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    video[0].save(output_path, format='PNG')

                else:
                    print("error")
                    assert 0

        # assert 0     

def decode_train_pth(pth_dir, decode_dir, pipe):
    pth_files = sorted([f for f in os.listdir(pth_dir) if f.endswith('.pth')])

    pipe.load_models_to_device(['vae'])

    for pth_file in tqdm.tqdm(pth_files, desc="Checking .pth files"):
        pth_path = os.path.join(pth_dir, pth_file)
        data = torch.load(pth_path, map_location='cpu')

        keys_to_ignonre = ['vace_context', 'context', 'negative_context', 'noise_pred'
        ]

        # check all keys' shape and save to txt file 
        os.makedirs(os.path.dirname(os.path.join(decode_dir, pth_file)), exist_ok=True)
        with open(os.path.join(decode_dir, pth_file + '.txt'), 'w') as f:
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    f.write(f"{key}: {data[key].shape, type(data[key])}\n") 
                elif isinstance(data[key], str):
                    f.write(f"{key}: {data[key], type(data[key])}\n")
                elif isinstance(data[key], int):
                    f.write(f"{key}: {str(data[key]), type(data[key])}\n")
                elif isinstance(data[key], float):
                    f.write(f"{key}: {str(data[key]), type(data[key])}\n")
                elif isinstance(data[key], bool):
                    f.write(f"{key}: {str(data[key]), type(data[key])}\n")
                elif isinstance(data[key], torch.device):
                    f.write(f"{key}: {str(data[key]), type(data[key])}\n")
            f.write(f"total lens of keys: {len(data)}\n")
        # assert 0


        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if key in keys_to_ignonre:
                    print(f"Skipping {key} in {pth_file}...")
                    continue

                data[key] = data[key].to(dtype=pipe.torch_dtype, device=pipe.device)
                print(f"************* Decoding {key} in {pth_file} *************")
                # continue

                # save image 
                output_path_image = os.path.join(decode_dir, pth_file, f"{key}.png")
                os.makedirs(os.path.dirname(output_path_image), exist_ok=True)
                image = pipe.vae.decode(data[key][:,:,0,:,:].unsqueeze(2), device=pipe.device, tiled=True, tile_size=(30, 52), tile_stride=(15, 26))
                image = pipe.vae_output_to_video(image)
                image[0].save(output_path_image, format='PNG')
            
                # save video 
                real_video = data[key][:,:,1:,:,:]
                # print(real_video.shape); assert 0 # torch.Size([1, 16, 13, 60, 104])
                real_video = pipe.vae.decode(real_video, device=pipe.device, tiled=True, tile_size=(30, 52), tile_stride=(15, 26))
                video = pipe.vae_output_to_video(real_video)

                output_path = os.path.join(decode_dir, pth_file, f"{key}.mp4")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                save_video(video, output_path, fps=15, quality=5)
            

def run_inference(checkpoint_path, args):
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
    pipe.enable_vram_management()

    ### add torch compile
    pipe.vae = torch.compile(pipe.vae, mode="default")  # 编译 VAE 模块
    pipe.dit = torch.compile(pipe.dit, mode="default")  # 编译 DIT 模块
    pipe.text_encoder = torch.compile(pipe.text_encoder, mode="default")  # 编译文本编码器

    # pth_dir = '/opt/data/private/yyx/data/OpenVidHD/train_pth_v2'
    # decode_dir = '/opt/data/private/yyx/data/OpenVidHD/train_pth_v2_decoded'
    # pth_dir = '/opt/data/private/yyx/code/DiffSynth-Studio-main/exp/train/sr_vacefull_datapth_cjy'
    # decode_dir = '/opt/data/private/yyx/code/DiffSynth-Studio-main/exp/train/sr_vacefull_datapth_cjy/decoded'
    pth_dir = '/opt/data/private/yyx/code/DiffSynth-Studio-main/exp/train/sr_vacefull_datapth_cjy'
    decode_dir = '/opt/data/private/yyx/code/DiffSynth-Studio-main/exp/train/sr_vacefull_datapth_cjy/decoded'
    
    # check_pth_keys(pth_dir)
    decode_pth_keys(pth_dir, decode_dir, pipe)

    # pth_dir = './tmp_debug'
    # decode_dir = './tmp_debug_decoded'
    # decode_train_pth(pth_dir, decode_dir, pipe)



def extract_epoch_number(filename):
    match = re.search(r'epoch-(\d+)', filename)
    return int(match.group(1)) if match else -1
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffSynth auto checkpoint inference")
    parser.add_argument("--checkpoint", type=str, default='models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors', help="Checkpoint path (file or folder)")
    parser.add_argument("--winputvideo",default=False,action="store_true",help="Whether to use SwanLab logger.",)
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if os.path.isfile(ckpt_path):

        run_inference(ckpt_path, args)


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
            run_inference(ckpt, args)
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")
