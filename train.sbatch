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

NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}  # 默认值为 1
echo "Detected $NUM_GPUS GPUs from SLURM"

EXPERIMENT=1  # 控制跑哪个实验（改这里即可）

case "$EXPERIMENT" in
  1)
    echo "vace original with data pt"

    # CUDA_VISIBLE_DEVICES=0,1,2,3 \
    accelerate launch --mixed_precision=bf16 \
      --main_process_port 20000 \
      --num_processes 4 \
      --config_file ./examples/wanvideo/model_training/full/accelerate_config_1point3B.yaml\
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
      --resume_from_checkpoint ./exp/train/Wan2.1-VACE-1.3B_full_4gpu_datapth/epoch-506.safetensors \
      --output_path "./exp/train/Wan2.1-VACE-1.3B_full_4gpu_datapth" \
      --trainable_models "vace" \
      --extra_inputs "vace_video,vace_reference_image" \
      --use_gradient_checkpointing_offload --use_swanlab --swanlab_mode local --use_data_pt exp/train/data_process
    ;;
  2)
    echo "data preprocess"

    # CUDA_VISIBLE_DEVICES=4 \
    python examples/wanvideo/model_training/train.py \
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
      --output_path "./exp/train/data_process" \
      --trainable_models "vace" \
      --extra_inputs "vace_video,vace_reference_image" \
      --use_gradient_checkpointing_offload --use_swanlab --swanlab_mode disabled --data_process
    ;;
  3)
    echo "sr vace"

    # CUDA_VISIBLE_DEVICES=4 \
    accelerate launch --mixed_precision=bf16 \
      --num_processes 1 \
      --main_process_port 20003 \
      --config_file ./examples/wanvideo/model_training/full/accelerate_config_1point3B.yaml\
      examples/wanvideo/model_training/train.py \
      --dataset_base_path ../../data/OpenVidHD \
      --dataset_metadata_path ../../data/OpenVidHD/data_csv/OpenVidHD.csv \
      --degradation_config_path config/degradations.yaml \
      --data_file_keys "video,vace_video,vace_reference_image" \
      --height 480 \
      --width 832 \
      --num_frames 49 \
      --dataset_repeat 2 \
      --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
      --learning_rate 1e-4 \
      --num_epochs 1000000000000000 \
      --remove_prefix_in_ckpt "pipe.vace." \
      --output_path "./exp/train/sr_vace" \
      --trainable_models "vace" \
      --extra_inputs "vace_video,vace_reference_image,degradation_kernels,degradation_params" \
      --use_gradient_checkpointing_offload --use_swanlab --swanlab_mode local --is_sr
    ;;
  4)
    echo "sr vace datapth"

    # CUDA_VISIBLE_DEVICES=0,1,2,3 \
    accelerate launch --mixed_precision=bf16 \
      --num_processes 4 \
      --main_process_port 20003 \
      --config_file ./examples/wanvideo/model_training/full/accelerate_config_1point3B.yaml\
      examples/wanvideo/model_training/train.py \
      --dataset_base_path ../../data/OpenVidHD \
      --dataset_metadata_path ../../data/OpenVidHD/data_csv/OpenVidHD.csv \
      --degradation_config_path config/degradations.yaml \
      --data_file_keys "video,vace_video,vace_reference_image" \
      --height 480 \
      --width 832 \
      --num_frames 49 \
      --dataset_repeat 2 \
      --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
      --learning_rate 1e-4 \
      --num_epochs 1000000000000000 \
      --remove_prefix_in_ckpt "pipe.vace." \
      --output_path "./exp/train/sr_vace_datapth" \
      --trainable_models "vace" \
      --extra_inputs "vace_video,vace_reference_image,degradation_kernels,degradation_params" \
      --use_gradient_checkpointing_offload --use_swanlab --swanlab_mode local --is_sr --use_data_pt ../../data/OpenVidHD/train_pth_cleaned

  5)
    echo "sr data preprocess"

    CUDA_VISIBLE_DEVICES=4 python examples/wanvideo/model_training/train.py \
      --dataset_base_path ../../data/OpenVidHD \
      --dataset_metadata_path ../../data/OpenVidHD/data_csv/OpenVidHD.csv \
      --degradation_config_path config/degradations.yaml \
      --data_file_keys "video,vace_video,vace_reference_image" \
      --height 480 \
      --width 832 \
      --num_frames 49 \
      --dataset_repeat 2 \
      --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
      --learning_rate 1e-4 \
      --num_epochs 1000000000000000 \
      --remove_prefix_in_ckpt "pipe.vace." \
      --output_path "/opt/data/private/yyx/data/OpenVidHD/train_pth" \
      --trainable_models "vace" \
      --extra_inputs "vace_video,vace_reference_image,degradation_kernels,degradation_params" \
      --use_gradient_checkpointing_offload --use_swanlab --swanlab_mode disabled --is_sr --data_process
  *)
    echo "Unknown experiment number: $EXPERIMENT"
    exit 1
    ;;
esac

