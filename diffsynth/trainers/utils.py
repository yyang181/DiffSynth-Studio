import imageio, os, torch, warnings, torchvision, argparse
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image, ImageFilter
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
import swanlab
import glob
import cv2
from torchvision.transforms import v2
import os
import numpy as np
from einops import rearrange
import random
import pickle
from io import BytesIO
import torch.nn.functional as F
import torchvision.transforms as T
import  torch.nn  as nn
from decord import VideoReader, cpu
import traceback


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

# BasicSR
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
import decord

os.environ["TOKENIZERS_PARALLELISM"] = "false"
decord.bridge.set_bridge('torch')

class SR_VideoDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        # metadata = pd.read_csv(metadata_path)
        # self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        # self.text = metadata["text"].to_list()
        self.max_num_frames = 81
        self.frame_interval = 2
        self.num_frames = args.num_frames
        self.height = args.height
        self.width = args.width
        self.is_i2v = False
        self.steps_per_epoch = 1
        self.image_file_extension=("jpg", "jpeg", "png", "webp")
        self.video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm")
        self.repeat = args.dataset_repeat

        self.SR_dataset = True
        self.sample_fps = self.frame_interval
        self.max_frames = self.max_num_frames
        self.misc_size = [self.height, self.width]
        self.video_list = []

        config_path = args.degradation_config_path
        ### real-esrgan settings
        # blur settings for the first degradation
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # print(config)
        degradation = config['degradation']
        self.degradation_params = config['degradation_params']

        self.blur_kernel_size = degradation['blur_kernel_size']
        self.kernel_list = degradation['kernel_list']
        self.kernel_prob = degradation['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = degradation['blur_sigma']
        self.betag_range = degradation['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = degradation['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = degradation['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = degradation['blur_kernel_size2']
        self.kernel_list2 = degradation['kernel_list2']
        self.kernel_prob2 = degradation['kernel_prob2']
        self.blur_sigma2 = degradation['blur_sigma2']
        self.betag_range2 = degradation['betag_range2']
        self.betap_range2 = degradation['betap_range2']
        self.sinc_prob2 = degradation['sinc_prob2']
        # print(self.sinc_prob2);assert 0
        
        # a final sinc filter
        self.final_sinc_prob = degradation['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1
        

        ### read csv file
        metadata_path = args.dataset_metadata_path
        df = pd.read_csv(metadata_path)
        self.csv_caption_data = df

        # # text promt
        # video_path = '/gemini/code/yyx/data/OpenVidHD/videos/3vuHE5QEn1k_18_50to186.mp4'
        # video_name = os.path.basename(video_path)
        # caption = df[df['video'] == video_name]['caption'].values
        # print(caption);assert 0

        # read all mp4 files
        abs_base_path = os.path.abspath(args.dataset_base_path)
        self.pose_dir = os.path.join(abs_base_path, "videos")
        search_pattern = os.path.join(abs_base_path, "videos", '*.mp4')
        mp4_files = sorted([mp4 for mp4 in glob.glob(search_pattern, recursive=True) if not mp4.startswith(".")])
        mp4_files_absolute = [os.path.abspath(mp4_file) for mp4_file in mp4_files]
        file_dict = {os.path.basename(mp4_file): mp4_file for mp4_file in mp4_files_absolute}

        # # 读取 CSV 文件
        # video_list = []
        # caption_list = []
        # df = pd.read_csv(metadata_path)
        # len_df = len(df)
        # for index, row in df.iterrows():
        #     # video_list.append(row["video"])
        #     video_list.append(file_dict[row["video"]])
        #     caption_list.append(row["caption"])
        # self.video_list = video_list
        # self.caption_list = caption_list
        # print('len_df:', len_df, 'len_videos:', len(video_list), 'len_captions:', len(caption_list))
        # assert len(video_list) == len(caption_list)


        # self.pose_dir = os.path.join(base_path, "videos")
        # file_list = [mp4 for mp4 in os.listdir(self.pose_dir) if mp4.endswith(".mp4") and not mp4.startswith(".")]
        # print("!!! all dataset length (OpenVidHD): ", len(file_list))
        # # 
        # for iii_index in file_list:
        #     self.video_list.append(os.path.join(self.pose_dir,iii_index))

        self.video_list = mp4_files

        self.use_pose = True
        print("!!! dataset length: ", len(self.video_list))

        # random.shuffle(self.video_list)
            
        self.frame_process = v2.Compose([
            # v2.CenterCrop(size=(height, width)), 
            v2.Resize(size=(self.height, self.width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def resize(self, image):
        width, height = image.size
        # 
        image = torchvision.transforms.functional.resize(
            image,
            (self.height, self.width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        # return torch.from_numpy(np.array(image))
        return image
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    

    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None

    def RealESRGAN_degradation(self):
        ### RealESRGAN degradation
        degradation_params = self.degradation_params
        first_degradation_seed = np.random.randint(2147483647)
        reseed(first_degradation_seed)
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        self.kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        second_degradation_seed = np.random.randint(2147483647)
        reseed(second_degradation_seed)
        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        self.kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        final_sinc_seed = np.random.randint(2147483647)
        reseed(final_sinc_seed)
        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            self.sinc_kernel = torch.FloatTensor(sinc_kernel).contiguous()
        else:
            self.sinc_kernel = self.pulse_tensor

        self.kernel = torch.FloatTensor(self.kernel).contiguous()
        self.kernel2 = torch.FloatTensor(self.kernel2).contiguous()

        degradation_kernels = {
            'kernel1': self.kernel,
            'kernel2': self.kernel2,
            'sinc_kernel': self.sinc_kernel,
            }
        return degradation_kernels

    # def __getitem__(self, data_id):
    def __getitem__(self, index):
        ### RealESRGAN degradation
        degradation_kernels = self.RealESRGAN_degradation()

        index = index % len(self.video_list)
        success=False
        for _try in range(5):
            try:
                if _try >0:
                    
                    index = random.randint(1,len(self.video_list))
                    index = index % len(self.video_list)
                
                clean = True
                path_dir = self.video_list[index]

                # get all frames
                # print(path_dir);assert 0 # /gemini/code/yyx/data/OpenVidHD/videos/3vuHE5QEn1k_18_50to186.mp4
                vr = VideoReader(path_dir)
                frames = vr.get_batch(range(len(vr)))  # 获取所有帧，返回的是 NDArray，形状为 [T, H, W, 3]
                
                # image = frames.permute(0, 3, 1, 2).contiguous() / 255.0

                del vr
                # print(frames.shape);assert 0 # (272, 1080, 1920, 3)

                frames_all = [Image.fromarray(frame.byte().cpu().numpy()) for frame in frames]
                dwpose_all = [Image.fromarray(frame.byte().cpu().numpy()) for frame in frames]

                ### video caption
                video_name = os.path.basename(path_dir)
                caption = self.csv_caption_data[self.csv_caption_data['video'] == video_name]['caption'].values[0]
                # print(caption);assert 0


                ### random sample fps
                stride = random.randint(1, self.sample_fps)  
                
                _total_frame_num = len(frames_all)
                cover_frame_num = (stride * self.max_frames)
                max_frames = self.max_frames
                if _total_frame_num < cover_frame_num + 1:
                    start_frame = 0
                    end_frame = _total_frame_num-1
                    stride = max((_total_frame_num//max_frames),1)
                    end_frame = min(stride*max_frames, _total_frame_num-1)
                else:
                    start_frame = random.randint(0, _total_frame_num-cover_frame_num)
                    end_frame = start_frame + cover_frame_num
                frame_list = []
                dwpose_list = []

                ### get reference frame
                random_ref = random.randint(0,_total_frame_num-1)
                # print('start_frame:', start_frame, 'end_frame:', end_frame, 'stride:', stride, '_total_frame_num', _total_frame_num, 'random_ref', random_ref)
                # # start_frame: 296 end_frame: 377 stride: 1 _total_frame_num 719 random_ref 228

                random_ref_frame = frames_all[random_ref]
                # print('random_ref_frame:', np.array(random_ref_frame).shape, np.min(random_ref_frame), np.max(random_ref_frame));assert 0 # random_ref_frame: (1080, 1920, 3) 0 255
 
                if random_ref_frame.mode != 'RGB' and not isinstance(random_ref_frame, torch.Tensor):
                    random_ref_frame = random_ref_frame.convert('RGB')
                random_ref_dwpose = dwpose_all[random_ref] if self.SR_dataset else Image.open(BytesIO(dwpose_all[i_key]))
                # print('random_ref_frame:', np.array(random_ref_frame).shape, np.min(random_ref_frame), np.max(random_ref_frame));assert 0 # random_ref_frame: (1080, 604, 3) 0 255
                
                ### read data
                first_frame = None
                for i_index in range(start_frame, end_frame, stride):
                    if self.SR_dataset:
                        i_frame = frames_all[i_index]
                        if i_frame.mode != 'RGB' and not isinstance(i_frame, torch.Tensor):
                            i_frame = i_frame.convert('RGB')
                        i_dwpose = dwpose_all[i_index]
                    else:
                        i_key = list(frames_all.keys())[i_index]
                        i_frame = Image.open(BytesIO(frames_all[i_key]))
                        if i_frame.mode != 'RGB' and not isinstance(i_frame, torch.Tensor):
                            i_frame = i_frame.convert('RGB')
                        i_dwpose = Image.open(BytesIO(dwpose_all[i_key]))
                    
                    if first_frame is None:
                        first_frame=i_frame

                        frame_list.append(i_frame)
                        dwpose_list.append(i_dwpose)

                    else:
                        frame_list.append(i_frame)
                        dwpose_list.append(i_dwpose)

                if (end_frame-start_frame) < max_frames:
                    for _ in range(max_frames-(end_frame-start_frame)):
                        if self.SR_dataset:
                            i_frame = frames_all[end_frame-1]
                            if i_frame.mode != 'RGB' and not isinstance(i_frame, torch.Tensor):
                                i_frame = i_frame.convert('RGB')
                            i_dwpose = dwpose_all[end_frame-1]
                        else:
                            i_key = list(frames_all.keys())[end_frame-1]
                            
                            i_frame = Image.open(BytesIO(frames_all[i_key]))
                            if i_frame.mode != 'RGB' and not isinstance(i_frame, torch.Tensor):
                                i_frame = i_frame.convert('RGB')
                            i_dwpose = Image.open(BytesIO(dwpose_all[i_key]))
                        
                        frame_list.append(i_frame)
                        dwpose_list.append(i_dwpose)

                have_frames = len(frame_list)>0
                middle_indix = 0

                if have_frames:

                    l_hight = random_ref_frame.size[1]
                    l_width = random_ref_frame.size[0]

                    # random crop
                    x1 = random.randint(0, l_width//14)
                    x2 = random.randint(0, l_width//14)
                    y1 = random.randint(0, l_hight//14)
                    y2 = random.randint(0, l_hight//14)
                    
                    
                    random_ref_frame = random_ref_frame.crop((x1, y1,l_width-x2, l_hight-y2))
                    ref_frame = random_ref_frame 
                    # 
                    
                    random_ref_frame_tmp = torch.from_numpy(np.array(self.resize(random_ref_frame)))
                    random_ref_dwpose_tmp = torch.from_numpy(np.array(self.resize(random_ref_dwpose.crop((x1, y1,l_width-x2, l_hight-y2))))) # [3, 512, 320]
                    
                    video_data_tmp = torch.stack([self.frame_process(self.resize(ss.crop((x1, y1,l_width-x2, l_hight-y2)))) for ss in frame_list], dim=0) # self.transforms(frames)
                    dwpose_data_tmp = torch.stack([torch.from_numpy(np.array(self.resize(ss.crop((x1, y1,l_width-x2, l_hight-y2))))).permute(2,0,1) for ss in dwpose_list], dim=0)

                video_data = torch.zeros(self.max_frames, 3, self.misc_size[0], self.misc_size[1])
                dwpose_data = torch.zeros(self.max_frames, 3, self.misc_size[0], self.misc_size[1])
                
                if have_frames:
                    video_data[:len(frame_list), ...] = video_data_tmp      
                    
                    dwpose_data[:len(frame_list), ...] = dwpose_data_tmp
                    
                video_data = video_data.permute(1,0,2,3)
                dwpose_data = dwpose_data.permute(1,0,2,3)
                
                break
            except Exception as e:
                # 
                caption = "a person is dancing"
                # 
                video_data = torch.zeros(3, self.max_frames, self.misc_size[0], self.misc_size[1])  
                random_ref_frame_tmp = torch.zeros(self.misc_size[0], self.misc_size[1], 3)
                vit_image = torch.zeros(3,self.misc_size[0], self.misc_size[1])
                
                dwpose_data = torch.zeros(3, self.max_frames, self.misc_size[0], self.misc_size[1])  
                random_ref = 0
                # 
                random_ref_dwpose_data = torch.zeros(3, self.max_frames, self.misc_size[0], self.misc_size[1])  
                print('{} read video frame failed with error: {}'.format(path_dir, e))
                traceback.print_exc()
                continue


        text = caption 
        path = path_dir 

        if self.is_i2v:
            video, first_frame = video_data, random_ref_frame_tmp
            # print(video.shape, torch.min(video), torch.max(video), first_frame.shape, torch.min(first_frame), torch.max(first_frame));assert 0
            # torch.Size([3, 81, 480, 720]) tensor(-1.) tensor(1.)  torch.Size([480, 720, 3]) tensor(0, dtype=torch.uint8) tensor(255, dtype=torch.uint8)

            data = {"text": text, "video": video, "path": path, "first_frame": first_frame, "dwpose_data": dwpose_data, "random_ref_dwpose_data": random_ref_dwpose_tmp, "degradation_kernels": degradation_kernels, "degradation_params": degradation_params}
        else:
            video, first_frame = video_data, random_ref_frame_tmp
            # data = {"text": text, "video": video, "path": path, "first_frame": first_frame, "dwpose_data": dwpose_data, "random_ref_dwpose_data": random_ref_dwpose_tmp, "degradation_kernels": degradation_kernels, "degradation_params": degradation_params}

            ### convert video from tensor to PIL.Image 
            video = (video + 1.0) / 2.0  # 将 [-1, 1] 映射到 [0, 1]
            video = video.permute(1, 0, 2, 3)  # [C, T, H, W] -> [T, C, H, W]
            video = [torchvision.transforms.functional.to_pil_image(frame) for frame in video]

            ### convert first_frame from tensor to PIL.Image
            # print(first_frame.shape, torch.min(first_frame), torch.max(first_frame));assert 0 # torch.Size([480, 832, 3]) tensor(0, dtype=torch.uint8) tensor(255, dtype=torch.uint8)
            first_frame = first_frame.permute(2, 0, 1)
            first_frame = torchvision.transforms.functional.to_pil_image(first_frame)

            data = {"prompt": text, "video": video, "vace_video": video, "vace_reference_image": first_frame, "degradation_kernels": degradation_kernels, "degradation_params": self.degradation_params}
        return data
    

    def __len__(self):
        return len(self.video_list) * self.repeat


class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
        else:
            metadata = pd.read_csv(metadata_path)
        self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        # print(self.data);assert 0 # [{'video': 'video1.mp4', 'prompt': 'from sunset to night, a small town, light, house, river', 'vace_video': 'video1_softedge.mp4', 'vace_reference_image': 'reference_image.png'}]
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    

    def load_video(self, file_path):
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat
    
class VideoDataset_pt(torch.utils.data.Dataset):
    def __init__(
        self,
        args=None,
    ):  
        self.pth_dir = os.path.abspath(args.use_data_pt) if args is not None else None
        self.pth_paths = [f for f in os.listdir(self.pth_dir) if f.endswith(".pth") and not f.startswith(".")]
        # sort by the number in the file name
        self.pth_paths.sort(key=lambda x: int(x.split(".")[0]))
        # print(self.pth_paths);assert 0 # ['data_cache/0.pth', 'data_cache/1.pth', ...]

        if 0:
            data_id = 1
            pth_name = self.pth_paths[data_id % len(self.pth_paths)]
            path = os.path.join(self.pth_dir, pth_name)
            input_dict = torch.load(path, map_location="cpu")
            if input_dict is None:
                warnings.warn(f"cannot load file {path}.")
                return None
            for key in input_dict.keys():
                print(key)

            print(f"Input video size: {input_dict['input_video'][0].size}")
            print(f"Input video num frames: {input_dict['num_frames']}")
            print(f"Input video height: {input_dict['height']}")
            print(f"Input video width: {input_dict['width']}")
            for extra_input in input_dict.keys():
                if extra_input not in ["input_video", "num_frames", "height", "width"]:
                    if isinstance(input_dict[extra_input], list):
                        print(f"Extra input {extra_input}: {[img.size for img in input_dict[extra_input]]}")
                    elif isinstance(input_dict[extra_input], torch.Tensor):
                        print(f"Extra input {extra_input}: {input_dict[extra_input].size()}")
                    else:
                        print(f"Extra input {extra_input}: {input_dict[extra_input]}")
            assert 0
            # Input video size: (832, 480)                                                                                 
            # Input video num frames: 49                                                                                   
            # Input video height: 480                                                                                                                                                                                                    
            # Input video width: 832                                                                                       
            # Extra input cfg_scale: 1                                                                                                                                                                                                   
            # Extra input tiled: False            
            # Extra input rand_device: cuda                                                                                                                                                                                              
            # Extra input use_gradient_checkpointing: True                                                                 
            # Extra input use_gradient_checkpointing_offload: True                                                                                                                                                                       
            # Extra input cfg_merge: False   
            # Extra input vace_scale: 1                                                                                                                                                                                                  
            # Extra input vace_video: [(832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (8
            # 32, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832,
            # 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480)]
            # Extra input vace_reference_image: <PIL.Image.Image image mode=RGB size=832x480 at 0x7F29FD9EC820>                                                                                                                          
            # Extra input noise: torch.Size([1, 16, 14, 60, 104])                                                          
            # Extra input latents: torch.Size([1, 16, 14, 60, 104]) 
            # Extra input input_latents: torch.Size([1, 16, 14, 60, 104])
            # Extra input vace_context: torch.Size([1, 96, 14, 60, 104])
            # Extra input prompt: from sunset to night, a small town, light, house, river
            # Extra input context: torch.Size([1, 512, 4096])


    def __getitem__(self, data_id):
        pth_name = self.pth_paths[data_id % len(self.pth_paths)]

        if self.pth_dir is not None:
            path = os.path.join(self.pth_dir, pth_name)
            input_dict = torch.load(path, map_location="cpu")
            if input_dict is None:
                warnings.warn(f"cannot load file {path}.")
                return None
        # print(input_dict);assert 0

        data = input_dict
        return data
    

    def __len__(self):
        return len(self.pth_paths)



class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        return model
    
    
    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict



class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        
    
    def on_step_end(self, accelerator, loss, step_id, epoch_id, scheduler=None):
        # Print the loss for the current step
        # This is where you can also log the loss to a file or a logging system
        # print(loss.item())
        # # add swanlab log 
        if accelerator.is_main_process:
            # swanlab.log({
            #     "train/loss": loss.item(),
            #     "lr": scheduler.get_last_lr()[0],
            #     "epoch": epoch_id,
            #     "step": step_id,
            # })
            accelerator.log({
                "train/loss": loss.item(),
                "lr": scheduler.get_last_lr()[0],
                "epoch": epoch_id,
                "step": step_id,
            })
    
    
    def on_epoch_end(self, accelerator, model, epoch_id):
        save_checkpoint_interval = 100
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and epoch_id % save_checkpoint_interval == 0:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)



def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    use_data_pt: str = None,
    args=None,
):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size, collate_fn=lambda x: x[0])
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, log_with="swanlab" if args.use_swanlab else None,)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    if args.use_swanlab and accelerator.is_main_process:
        import swanlab
        exp_name = args.output_path.split(os.sep)[-1]
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))

        # Initialise your swanlab experiment, passing swanlab parameters and any config information
        accelerator.init_trackers(
            project_name="DiffSynth-Studio",
            config=swanlab_config,
            init_kwargs={"swanlab": {"experiment_name": exp_name, "mode": args.swanlab_mode}},
            )

    step_id = 0
    for epoch_id in range(num_epochs):
        with tqdm(dataloader, desc=f"Epoch {epoch_id + 1}/{num_epochs}, Step {step_id}") as pbar:
            for data_id, data in enumerate(pbar):
                step_id += 1
                # with accelerator.accumulate(model):
                #     optimizer.zero_grad()
                #     loss = model(data) if use_data_pt is None else model(data, inputs=data)
                #     accelerator.backward(loss)
                #     optimizer.step()
                #     model_logger.on_step_end(accelerator, loss, step_id, epoch_id, scheduler=scheduler)
                #     scheduler.step()
                optimizer.zero_grad()
                loss = model(data) if use_data_pt is None else model(data, inputs=data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, loss, step_id, epoch_id, scheduler=scheduler)
                scheduler.step()

                # 更新 tqdm 的显示内容
                pbar.set_postfix(loss=loss.item())
        
        model_logger.on_epoch_end(accelerator, model, epoch_id)
    
    accelerator.end_training()



def launch_data_process_task(model: DiffusionTrainingModule, dataset, output_path="./models"):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0])
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    # os.makedirs(os.path.join(output_path, "data_cache"), exist_ok=True)
    for data_id, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = model.forward_preprocess(data)
            inputs = {key: inputs[key] for key in model.model_input_keys if key in inputs}
            os.makedirs(output_path, exist_ok=True)
            torch.save(inputs, os.path.join(output_path, f"{data_id}.pth"))



def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to resume from checkpoint.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--use_swanlab",default=False,action="store_true",help="Whether to use SwanLab logger.",)
    parser.add_argument("--swanlab_mode", default=None, help="SwanLab mode (cloud or local).",)
    parser.add_argument("--data_process",default=False,action="store_true",help="Whether to use SwanLab logger.",)
    parser.add_argument("--is_sr",default=False,action="store_true",help="Whether to use SwanLab logger.",)
    parser.add_argument("--use_data_pt",default=None,help="Whether to use SwanLab logger.",)
    parser.add_argument("--degradation_config_path", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    return parser

