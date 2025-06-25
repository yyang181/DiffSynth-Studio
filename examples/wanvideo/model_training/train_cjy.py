import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
import sys
sys.path.append('/gemini/code/cjy/DiffSynth-Studio-main')
from diffsynth import ModelManager, load_state_dict, save_video
from diffsynth.pipelines.wan_video_v1 import WanVideoPipeline_v1 as WanVideoPipeline
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
from transformers import get_linear_schedule_with_warmup
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from basicsr.data.realbasicvsr_dataset_video import RealVSRRecurrentDataset as TextVideoDataset
from torchvision.transforms import v2
from glob import glob
import yaml


class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        print(tiled)
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
       
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
    def test_step(self, batch, batch_idx):
        text, video, lq_video, path = batch["text"][0], batch["video"], batch["lq_video"], batch["path"][0]
        # print(batch["video"])
        self.pipe.device = self.device
        if isinstance(video, torch.Tensor):
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            neg_prompt_emb = self.pipe.encode_prompt('')
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            # video
            lq_video = lq_video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            lq_latents = self.pipe.encode_video(lq_video, **self.tiler_kwargs)[0]

            # image
            if "first_frame" in batch:
                first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                _, _, num_frames, height, width = video.shape
                image_emb = self.pipe.encode_image(first_frame, num_frames, height, width)
            else:
                image_emb = {}
            data = {"latents": latents, "lq_latents": lq_latents,"prompt_emb": prompt_emb, "neg_prompt_emb": neg_prompt_emb, "image_emb": image_emb}
            # print(path, text)
            torch.save(data, path + ".tensors.pth")
            print(path)


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, steps_per_epoch):
        video_49_frames_HQ_path = sorted(glob(os.path.join('/gemini/code/cjy/High_quality_training_data/Video_Musiq_Maniqa_sort_level_I_Tensor_49_frames', '**', '*.pth'), recursive=True))

        self.path = video_49_frames_HQ_path

        print(len(self.path), "tensors cached in metadata.")
        assert len(self.path) > 0
        
        self.steps_per_epoch = steps_per_epoch


    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path)  # For fixed seed.
        path = self.path[data_id]
        data = torch.load(path, weights_only=True, map_location="cpu")


        return data

    

    def __len__(self):
        return self.steps_per_epoch


class val_TensorDataset(torch.utils.data.Dataset):
    def __init__(self, steps_per_epoch):
        self.path = sorted(glob(os.path.join('/gemini/code/cjy/Validation_dataset/VideoLQ_for_train_vailidation/sample_pth', '**', '*.pth'), recursive=True))
        print(len(self.path), "validation_data.")
        assert len(self.path) > 0
        
        self.steps_per_epoch = steps_per_epoch


    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path) # For fixed seed.
        path = self.path[data_id]
        data = torch.load(path, weights_only=True, map_location="cpu")
        return data
    

    def __len__(self):
        return self.steps_per_epoch



class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        total_steps = 20000,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.denoising_model().enable_lq_condition()
        
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)
        
        self.learning_rate = learning_rate
        self.warmup_steps = 500
        self.total_steps = total_steps
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    

    def training_step(self, batch, batch_idx):
        # Data
        latents = batch["latents"].to(self.device)
        lq_latents = batch["lq_latents"].to(self.device)
        
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][:,0,:,:].to(self.device)
        image_emb = batch["image_emb"]
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)

        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, lq_input = lq_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, on_step=True, on_epoch=True)
        return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)

        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.warmup_steps,
        #     num_training_steps=self.total_steps,
        # )
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)


    def validation_step(self, batch, batch_idx):
        # 验证步骤
        if self.global_step == 0:
            return {"val_loss": 0} 
        latents = batch["latents"].to(self.device)
        lq_latents = batch["lq_latents"].to(self.device)

        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)

        neg_prompt_emb = batch["neg_prompt_emb"]
        neg_prompt_emb["context"] = neg_prompt_emb["context"][0].to(self.device)

        fake_latents = self.pipe.test(prompt_emb_posi = prompt_emb, prompt_emb_nega=neg_prompt_emb, lq_latents=lq_latents, num_frames=49, num_inference_steps=20, cfg_scale = 4, device = self.device)
        fake_latents_path = os.path.join(self.trainer.default_root_dir, f"iter_{self.global_step}_{batch_idx}.pth")
        data = {"fake_latents": fake_latents}
        torch.save(data, fake_latents_path)
        loss = torch.nn.functional.mse_loss(fake_latents.float(), latents.float())
        print(f"iter_{self.global_step}_{batch_idx}_val_loss {loss}")
        self.log('val_loss', loss)
        return {"val_loss": loss}


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train", 'data_decode'],
        help="Task. `data_process` or `train` or `data_decode`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    args = parser.parse_args()
    return args


def data_process(args):

    config_file = "/gemini/code/cjy/DiffSynth-Studio-main/examples/wanvideo/train_config.yaml"
    with open(config_file, "r") as f:
        opt = yaml.safe_load(f)

    video_path = "/gemini/code/cjy/High_quality_training_data/high_quality/Video_Musiq_Maniqa_sort_level_I"
    json_path = "/gemini/code/cjy/High_quality_training_data/high_quality/Video_Musiq_Maniqa_sort_level_I_caption"
    dataset = TextVideoDataset(opt, video_path, json_path, num_frame=49, img_size=(512, 832))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)
    



def train(args):
    dataset = TensorDataset(
        steps_per_epoch=args.steps_per_epoch,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=12,
        num_workers=args.dataloader_num_workers
    )

    val_dataset = val_TensorDataset(
        steps_per_epoch=2,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )

    model = LightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        total_steps=100000,
        train_architecture=args.train_architecture,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
    )
    # if args.use_swanlab:
    #     from swanlab.integration.pytorch_lightning import SwanLabLogger
    #     swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
    #     swanlab_config.update(vars(args))
    #     swanlab_logger = SwanLabLogger(
    #         project="wan", 
    #         name="wan",
    #         config=swanlab_config,
    #         mode=args.swanlab_mode,
    #         logdir=os.path.join(args.output_path, "swanlog"),
    #     )
    #     logger = [swanlab_logger]
    # else:
    tensor_logger = TensorBoardLogger(save_dir=os.path.join(args.output_path, "tensorboardlog"),name="v1")
    logger = [tensor_logger]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1, every_n_train_steps=2000)],
        logger=logger,
        log_every_n_steps=1,
        val_check_interval = 1000,
    )
    trainer.fit(model, dataloader, val_dataloader)


if __name__ == '__main__':
    args = parse_args()
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)
