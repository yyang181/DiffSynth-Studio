#!/bin/bash
#SBATCH --job-name=wan_vace_train
#SBATCH --output=logs/wan_vace_train_%j.out
#SBATCH --error=logs/wan_vace_train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=ai

# 激活 conda 环境
/HOME/paratera_xy/pxy545/miniconda3/bin/conda init             # 确保 conda 命令可用（必要）
conda activate diffsynth           # 替换为你的实际 conda 环境名

# （可选）显示当前环境和 CUDA 信息，方便调试
echo "Conda env: $CONDA_DEFAULT_ENV"
which python
nvidia-smi

# 训练命令
accelerate launch --mixed_precision=bf16 \
  --main_process_port=20000 \
  --config_file ./examples/wanvideo/model_training/full/accelerate_config_1point3B.yaml \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata_vace.csv \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 1000000000000000 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./exp/train/Wan2.1-VACE-1.3B_full" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,vace_reference_image" \
  --use_gradient_checkpointing_offload \
  --use_swanlab \
  --swanlab_mode local
