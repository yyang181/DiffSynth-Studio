import torch, os, json
from diffsynth.pipelines.wan_video_new_v1_vacefull import ModelConfig
from diffsynth.pipelines.wan_video_new_v1_vacefull import WanVideoPipeline_v1_vacefull as WanVideoPipeline 
from diffsynth.trainers.utils import DiffusionTrainingModule, VideoDataset, VideoDataset_pt, SR_VideoDataset, ModelLogger, launch_training_task, wan_parser, launch_data_process_task
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# realesrgan 
import yaml
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from utils.seed import reseed
from utils.util import to_device, to_numpy, to_item
import math
from PIL import Image, ImageFilter
import numpy as np
from torchvision import transforms
import random
from einops import rearrange
import torch.nn.functional as F
import torchvision.transforms as T

# BasicSR
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop

from diffsynth import load_state_dict

# 定义 img2tensor 函数
def img2tensor(img):
    """
    将PIL图像转换为torch tensor。
    假设输入的图像是PIL格式。
    """
    transform = transforms.ToTensor()
    return transform(img)

@torch.no_grad()
def do_degredation(video_data, degradation_kernels, opt, dtype=torch.float, reduce_temporal_vars=True):
    # input gt: B T C H W [-1, 1]

    ### if video_data is PIL.image or numpy array, convert it to tensor
    flag_return_pil = False
    if isinstance(video_data, (list, tuple)):
        if isinstance(video_data[0], (Image.Image,np.ndarray)):
            # list of PIL images
            # video_data = [torch.stack([img2tensor(img).to(dtype) for img in imgs], dim=1) for imgs in video_data]   
            video_tensor = torch.stack([img2tensor(img) for img in video_data], dim=1)  # [C, T, H, W]
            # print(f"video_tensor shape: {video_tensor.shape}, dtype: {video_tensor.dtype}, device: {video_tensor.device}, min: {torch.min(video_tensor)}, max: {torch.max(video_tensor)}")  # Debugging line to check video_tensor
            # assert 0 # video_tensor shape: torch.Size([3, 81, 480, 832]), dtype: torch.float32, device: cpu , min: 0.0, max: 1.0 
            video_data = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [B, T, C, H, W]
            video_data = video_data * 2.0 - 1.0  # Normalize to [-1, 1]
            video_data = video_data.to(device='cuda' if torch.cuda.is_available() else 'cpu', dtype=dtype)  # Move to GPU if available
            flag_return_pil = True
        else:
            raise TypeError(f"Unsupported video_data type in list: {type(video_data[0])}. Expected PIL.Image or numpy.ndarray.")
    elif isinstance(video_data, torch.Tensor):
        pass
    else:
        raise TypeError(f"Unsupported video_data type: {type(video_data)}. Expected list, tuple, or torch.Tensor.")

    # print(f"video_data type: {type(video_data)}, shape: {video_data.shape}, dtype: {video_data.dtype}, device: {video_data.device}, min: {video_data.min()}, max: {video_data.max()}") # Debugging line to check video_data
    # assert 0 # video_data type: <class 'torch.Tensor'>, shape: torch.Size([1, 81, 3, 480, 832]), dtype: torch.float32, device: cpu video_data min: -1.0, max: 1.0, mean: -0.5278885960578918 

    kernel1, kernel2, sinc_kernel = degradation_kernels['kernel1'], degradation_kernels['kernel2'], degradation_kernels['sinc_kernel']

    gt = video_data.float()
    gt = gt * 0.5 + 0.5
    b, t, c, ori_h, ori_w = gt.size()
    
    # usm_sharpener only support 4D tensor 
    gt4D = gt.reshape(b, t*c, ori_h, ori_w)

    # gt_usm4D = usm_sharpener(gt4D)
    gt_usm4D = gt4D

    kernel1 = kernel1.to(dtype)
    kernel2 = kernel2.to(dtype)
    sinc_kernel = sinc_kernel.to(dtype)

    # 1. 清理不再使用的变量（减少显存占用）
    if reduce_temporal_vars:
        del gt4D  
        torch.cuda.empty_cache()

    bt_seed = np.random.randint(2147483647)
    reseed(bt_seed)
    # ----------------------- The first degradation process ----------------------- #
    # blur
    # out = filter2D(gt_usm, kernel1)
    out = filter2D(gt_usm4D, kernel1).to(dtype)
    # random resize
    updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, opt['resize_range'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(opt['resize_range'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    # add noise
    gray_noise_prob = opt['gray_noise_prob']
    out = out.reshape(b*t, c, out.size()[2], out.size()[3])
    if np.random.uniform() < opt['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(
            out, sigma_range=opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=opt['poisson_scale_range'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)
    if 0:        
        # JPEG compression
        # shape handle, jpeger need B 3 H W to convert Y Cb Cr
        out = out.reshape(b*t, c, out.size()[2], out.size()[3])
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = jpeger(out, quality=jpeg_p)
    else:
        out = out.reshape(b*t, c, out.size()[2], out.size()[3])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts

    if reduce_temporal_vars:
        # 释放 kernel1 显存
        del kernel1
        torch.cuda.empty_cache()

    # ----------------------- The second degradation process ----------------------- #
    # blur
    if np.random.uniform() < opt['second_blur_prob']:
        out = out.reshape(b, t*c, out.size()[2], out.size()[3])
        out = filter2D(out, kernel2).to(dtype)
    # random resize
    updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob2'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, opt['resize_range2'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(opt['resize_range2'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
        out, size=(int(ori_h / opt['scale'] * scale), int(ori_w / opt['scale'] * scale)), mode=mode)
    # add noise
    gray_noise_prob = opt['gray_noise_prob2']
    out = out.reshape(b*t, c, out.size()[2], out.size()[3])
    if np.random.uniform() < opt['gaussian_noise_prob2']:
        out = random_add_gaussian_noise_pt(
            out, sigma_range=opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=opt['poisson_scale_range2'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)

    if reduce_temporal_vars:
        # 释放 kernel2 显存
        del kernel2
        torch.cuda.empty_cache()

    # JPEG compression + the final sinc filter
    # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
    # as one operation.
    # We consider two orders:
    #   1. [resize back + sinc filter] + JPEG compression
    #   2. JPEG compression + [resize back + sinc filter]
    # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
    if np.random.uniform() < 0.5:
        # resize back + the final sinc filter
        out = out.reshape(b, t*c, out.size()[2], out.size()[3])
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=mode)
        out = filter2D(out, sinc_kernel).to(dtype)

        if 0:
            # JPEG compression
            out = out.reshape(b*t, c, out.size()[2], out.size()[3])
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
        else:
            out = out.reshape(b*t, c, out.size()[2], out.size()[3])
            out = torch.clamp(out, 0, 1)
    else:
        if 0:
            # JPEG compression
            out = out.reshape(b*t, c, out.size()[2], out.size()[3])
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
        else:
            out = out.reshape(b*t, c, out.size()[2], out.size()[3])
            out = torch.clamp(out, 0, 1)

        # resize back + the final snc filter
        out = out.reshape(b, t*c, out.size()[2], out.size()[3])
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=mode)
        out = filter2D(out, sinc_kernel).to(dtype)

    if reduce_temporal_vars:
        # 释放 sinc_kernel 显存
        del sinc_kernel
        torch.cuda.empty_cache()

    # clamp and round
    lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    lq = lq.reshape(b, t*c, lq.size()[2], lq.size()[3])

    # # random crop
    # gt_size = opt['gt_size']
    # (gt, gt_usm), lq = paired_random_crop([gt, gt_usm4D], lq, gt_size,
    #                                                         opt['scale'])
    gt_usm = gt_usm4D

    lq = lq.reshape(b, t, c, lq.size()[2], lq.size()[3])
    gt_usm = gt_usm.reshape(b, t, c, gt_usm.size()[2], gt_usm.size()[3])

    lq = lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

    # print(lq.size(), gt.size(), gt_usm.size());assert 0
    # torch.Size([2, 8, 3, 224, 224]) torch.Size([2, 8, 3, 448, 448]) torch.Size([2, 8, 3, 448, 448])

    if reduce_temporal_vars:
        # 释放不再使用的变量
        del gt_usm4D, out
        torch.cuda.empty_cache()

    lq = F.interpolate(lq.reshape(b*t, c, lq.size()[3], lq.size()[4]), (gt_usm.size()[3], gt_usm.size()[4]), mode='bicubic', align_corners=False)
    lq = lq.reshape(b, t, c, lq.size()[2], lq.size()[3])

    if not flag_return_pil:
        lq = lq * 2.0  - 1.0
        lq = torch.clamp(lq, -1, 1)

        # batch['mp4_ori'] = gt.to(dtype)
        lq_video_data = lq.to(dtype)
        # batch['mp4'] = gt_usm.to(dtype) # true gt used for training 
    else:
        # 如果输入是 PIL 图像或 numpy 数组，转换回 PIL 图像 lq: B T C H W
        lq_video_data = lq.permute(0, 1, 3, 4, 2).squeeze(0)  # [B, T, C, H, W] -> [T, H, W, C]
        lq_video_data = torch.clamp((lq_video_data * 255.0).round(), 0, 255)
        # print(lq_video_data.shape, torch.min(lq_video_data), torch.max(lq_video_data)); assert 0  # torch.Size([81, 480, 832, 3]) tensor(0., device='cuda:0') tensor(255., device='cuda:0')
        
        lq_video_data = [Image.fromarray((img.cpu().numpy()).astype(np.uint8)) for img in lq_video_data]

    if reduce_temporal_vars:
        # 释放最终变量
        del gt, lq, gt_usm
        torch.cuda.empty_cache()

    return lq_video_data


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        resume_from_checkpoint=None,
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]

        # print(f"Loading models: {model_configs}");assert 0
        if resume_from_checkpoint is None: 
            self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
            
            # add vacefull parameters
            if hasattr(self.pipe.dit, 'vacefull_patch_embedding'):
                pass
            else:
                print("🔍 VACEFull patch embedding not found in the model, initilize it.")
                self.pipe.dit.enable_vacefull_condition()

        else:
            self.pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cpu",
                model_configs=[
                    ModelConfig(path=[resume_from_checkpoint], offload_device=None),
                    ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device=None),
                    ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device=None),
                ],
            )

            ### add vacefull parameters
            self.pipe.dit.enable_vacefull_condition()
            print(f"🔍 Loading vacefull_patch_embedding weights from resume ckpt: {resume_from_checkpoint}")
            dit_state_dict = load_state_dict(resume_from_checkpoint)
            vacefull_state_dict = {}
            for key in dit_state_dict.keys():
                if 'vacefull_patch_embedding.' in key:
                    vacefull_state_dict[key.split("vacefull_patch_embedding.")[1]] = dit_state_dict[key]
            self.pipe.dit.vacefull_patch_embedding.load_state_dict(vacefull_state_dict, strict=True)
            


        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank
            )
            setattr(self.pipe, lora_base_model, model)
            
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        
        # self.model_input_keys = ['input_video', 'height', 'width', 'num_frames', 'cfg_scale', 'tiled', 'rand_device', 'use_gradient_checkpointing', 'use_gradient_checkpointing_offload', 'cfg_merge', 'vace_scale', 'vace_video', 'vace_reference_image', 'noise', 'latents', 'input_latents', 'vace_context', 'prompt', 'context']
        self.model_input_keys = ['height', 'width', 'num_frames', 'cfg_scale', 'tiled', 'rand_device', 'use_gradient_checkpointing', 'use_gradient_checkpointing_offload', 'cfg_merge', 'vace_scale', 'noise', 'latents', 'input_latents', 'vace_context', 'vace_video_latent', 'vace_reference_latent', 'prompt', 'context', 'negative_context']
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        if 0:
            print(f"Input video size: {inputs_shared['input_video'][0].size}")
            print(f"Input video num frames: {inputs_shared['num_frames']}")
            print(f"Input video height: {inputs_shared['height']}")
            print(f"Input video width: {inputs_shared['width']}")
            for extra_input in self.extra_inputs:
                # check if extra_input is image or list of images
                if extra_input in inputs_shared:
                    if isinstance(inputs_shared[extra_input], list):
                        print(f"Extra input {extra_input}: {[img.size for img in inputs_shared[extra_input]]}")
                    elif isinstance(inputs_shared[extra_input], torch.Tensor):
                        print(f"Extra input {extra_input}: {inputs_shared[extra_input].size()}")
                    else:
                        print(f"Extra input {extra_input}: {inputs_shared[extra_input]}")
            assert 0

            # Input video size: (832, 480) 
            # Input video num frames: 49       
            # Input video height: 480 
            # Input video width: 832 
            # Extra input vace_video: [(832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480)]
            # Extra input vace_reference_image: <PIL.Image.Image image mode=RGB size=832x480 at 0x7FB27AF4A460>

            # print(inputs_shared.keys(), inputs_posi.keys(), inputs_nega.keys());assert 0 # 
            # dict_keys(['input_video', 'height', 'width', 'num_frames', 'cfg_scale', 'tiled', 'rand_device', 'use_gradient_checkpointing', 'use_gradient_checkpointing_offload', 'cfg_merge', 'vace_scale', 'vace_video', 'vace_reference_image']) dict_keys(['prompt']) dict_keys([])

        ### Degradation process
        # if inputs_shared['degradation_kernels'] is not None and inputs_shared['degradation_params'] is not None:
        if 'degradation_kernels' in inputs_shared and 'degradation_params' in inputs_shared:
            # print('video:', video.shape, video.dtype, video.device) # video: torch.Size([1, 3, 81, 480, 720]) torch.float32 cuda:0
            degradation_params = inputs_shared['degradation_params']
            # print(degradation_params);assert 0
            # {'scale': 4, 'resize_prob': [0.2, 0.7, 0.1], 'resize_range': [0.3, 1.5], 'gaussian_noise_prob': 0.5, 'noise_range': [1, 15], 'poisson_scale_range': [0.05, 2.0], 'gray_noise_pro
            # b': 0.4, 'jpeg_range': [60, 95], 'second_blur_prob': 0.5, 'resize_prob2': [0.3, 0.4, 0.3], 'resize_range2': [0.6, 1.2], 'gaussian_noise_prob2': 0.5, 'noise_range2': [1, 12], 'p
            # oisson_scale_range2': [0.05, 1.0], 'gray_noise_prob2': 0.4, 'jpeg_range2': [60, 100]}

            vace_video = do_degredation(inputs_shared['vace_video'], inputs_shared['degradation_kernels'], inputs_shared['degradation_params'], dtype=torch.float) # video.dtype
            inputs_shared['vace_video'] = vace_video

            if 0:
                print(vace_video)
                inputs_shared['vace_video'][0].save('vace_video.png')  # Save the first frame of the VACE video for debugging
                inputs_shared['input_video'][0].save('input_video.png')  # Save the first frame of the input video for debugging
                inputs_shared['vace_reference_image'].save('vace_reference_image.png')  # Save the reference image for debugging

                assert 0

        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)

        inputs_nega_update_key_name = {
            "negative_context": inputs_nega["context"] if "context" in inputs_nega else None,
        }

        # print(inputs_shared.keys(), inputs_posi.keys(), inputs_nega.keys());assert 0 # 
        # dict_keys(['input_video', 'height', 'width', 'num_frames', 'cfg_scale', 'tiled', 'rand_device', 'use_gradient_checkpointing', 'use_gradient_checkpointing_offload', 'cfg_merge', 'vace_scale', 'vace_video', 'vace_reference_image', 'noise', 'latents', 'input_latents', 'vace_context']) dict_keys(['prompt', 'context']) dict_keys(['context'])

        # return {**inputs_shared, **inputs_posi}
        return {**inputs_shared, **inputs_posi, **inputs_nega_update_key_name}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        # print(inputs.keys());assert 0  # Debugging line to check inputs and models

        if 0:
            ### TODO so ugly, need to fix the inputs
            ### fix pt_data_parameters
            extra_inputs = {
                "cfg_scale": 1,
                "tiled": False,
                "rand_device": self.pipe.device,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
                "cfg_merge": False,
                "vace_scale": 1,
            }
            for extra_input in extra_inputs:
                if extra_input in inputs:
                    pass
                else:
                    inputs[extra_input] = extra_inputs[extra_input]
                    # print(f"Warning: extra input {extra_input} not found in inputs, set to default value: {extra_inputs[extra_input]}")
            # print(f"Inputs keys: {inputs.keys()}")  # Debugging line to check inputs keys


        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()

    if not args.is_sr:
        dataset = VideoDataset(args=args) if args.use_data_pt is None else VideoDataset_pt(args=args)
    else:
        dataset = SR_VideoDataset(args=args) if args.use_data_pt is None else VideoDataset_pt(args=args)

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    assert not (args.use_data_pt is not None and args.data_process), \
        "You must choose one of --use_data_pt or --data_process, not both."
        
    if 0:
        ### add torch compile
        model.pipe.vae = torch.compile(model.pipe.vae, mode="default")  # 编译 VAE 模块
        model.pipe.dit = torch.compile(model.pipe.dit, mode="default")  # 编译 DIT 模块
        model.pipe.text_encoder = torch.compile(model.pipe.text_encoder, mode="default")  # 编译文本编码器
        model.pipe.vace = torch.compile(model.pipe.vace, mode="default")  # 编译文本编码器
    
    if args.data_process:
        # Launch data processing task
        launch_data_process_task(
            model, dataset, args.output_path
        )
    else:
        launch_training_task(
            dataset, model, model_logger, optimizer, scheduler,
            num_epochs=args.num_epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_data_pt=args.use_data_pt,
            args=args,
        )
