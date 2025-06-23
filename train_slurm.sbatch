#!/bin/bash
#SBATCH --job-name=$SLURM_JOB_NAME
#SBATCH --output=logs/%j.out   # 标准输出日志（%j 代表 Job ID）
#SBATCH --error=logs/%j.err    # 标准错误日志
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --partition=pxy545

# 初始化 conda（一次性）路径
source /HOME/paratera_xy/pxy545/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth           # 替换为你的实际 conda 环境名

NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}  # 默认值为 1
PORT=$((20000 + RANDOM % 1000))

echo "=== SLURM Job: $SLURM_JOB_NAME ==="
echo "Using Conda environment: $CONDA_DEFAULT_ENV"
echo "Start Time: $(date)"
echo "Using $NUM_GPUS GPUs"
which python
nvidia-smi

# 根据 job-name 选择执行逻辑
case "$SLURM_JOB_NAME" in

  vace_original)
    echo "Running: vace original with data pt"

    CMD=$(cat <<EOF
accelerate launch --mixed_precision=bf16 \
  --main_process_port ${PORT} \
  --num_processes ${NUM_GPUS} \
  --config_file ./examples/wanvideo/model_training/full/accelerate_config_1point3B.yaml \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata_vace.csv \
  --data_file_keys video,vace_video,vace_reference_image \
  --height 480 --width 832 --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth \
  --learning_rate 1e-4 --num_epochs 1000000000000000 \
  --remove_prefix_in_ckpt pipe.vace. \
  --resume_from_checkpoint ./exp/train/Wan2.1-VACE-1.3B_full_4gpu_datapth/epoch-506.safetensors \
  --output_path ./exp/train/Wan2.1-VACE-1.3B_full_${NUM_GPUS}gpu_datapth_resume506 \
  --trainable_models vace \
  --extra_inputs vace_video,vace_reference_image \
  --use_gradient_checkpointing_offload \
  --use_swanlab --swanlab_mode local \
  --use_data_pt exp/train/data_process
EOF
)
    ;;

  data_preprocess)
    echo "Running: data preprocess"

    CMD=$(cat <<EOF
python examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata_vace.csv \
  --data_file_keys video,vace_video,vace_reference_image \
  --height 480 --width 832 --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth \
  --learning_rate 1e-4 --num_epochs 1000000000000000 \
  --remove_prefix_in_ckpt pipe.vace. \
  --output_path ./exp/train/data_process \
  --trainable_models vace \
  --extra_inputs vace_video,vace_reference_image \
  --use_gradient_checkpointing_offload \
  --use_swanlab --swanlab_mode disabled \
  --data_process
EOF
)
    ;;

  sr_vace)
    echo "Running: sr vace"

    CMD=$(cat <<EOF
accelerate launch --mixed_precision=bf16 \
  --num_processes 1 \
  --main_process_port ${PORT} \
  --config_file ./examples/wanvideo/model_training/full/accelerate_config_1point3B.yaml \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path ../../data/OpenVidHD \
  --dataset_metadata_path ../../data/OpenVidHD/data_csv/OpenVidHD.csv \
  --degradation_config_path config/degradations.yaml \
  --data_file_keys video,vace_video,vace_reference_image \
  --height 480 --width 832 --num_frames 49 \
  --dataset_repeat 2 \
  --model_id_with_origin_paths Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth \
  --learning_rate 1e-4 --num_epochs 1000000000000000 \
  --remove_prefix_in_ckpt pipe.vace. \
  --output_path ./exp/train/sr_vace \
  --trainable_models vace \
  --extra_inputs vace_video,vace_reference_image,degradation_kernels,degradation_params \
  --use_gradient_checkpointing_offload \
  --use_swanlab --swanlab_mode local \
  --is_sr
EOF
)
    ;;

  sr_vace_datapth)
    echo "Running: sr vace datapth"

    CMD=$(cat <<EOF
accelerate launch --mixed_precision=bf16 \
  --num_processes ${NUM_GPUS} \
  --main_process_port ${PORT} \
  --config_file ./examples/wanvideo/model_training/full/accelerate_config_1point3B.yaml \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path ../../data/OpenVidHD \
  --dataset_metadata_path ../../data/OpenVidHD/data_csv/OpenVidHD.csv \
  --degradation_config_path config/degradations.yaml \
  --data_file_keys video,vace_video,vace_reference_image \
  --height 480 --width 832 --num_frames 49 \
  --dataset_repeat 2 \
  --model_id_with_origin_paths Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth \
  --learning_rate 1e-4 --num_epochs 1000000000000000 \
  --remove_prefix_in_ckpt pipe.vace. \
  --resume_from_checkpoint ./exp/train/Wan2.1-VACE-1.3B_full_4gpu_datapth_resume506/epoch-154.safetensors \
  --output_path ./exp/train/${SLURM_JOB_NAME}_${NUM_GPUS}gpu_resume506_resume154 \
  --trainable_models vace \
  --extra_inputs vace_video,vace_reference_image,degradation_kernels,degradation_params \
  --use_gradient_checkpointing_offload \
  --use_swanlab --swanlab_mode local \
  --is_sr \
  --use_data_pt ../../data/OpenVidHD/train_pth_cleaned
EOF
)
    ;;

  sr_data_preprocess)
    echo "Running: sr data preprocess"

    CMD=$(cat <<EOF
python examples/wanvideo/model_training/train.py \
  --dataset_base_path ../../data/OpenVidHD \
  --dataset_metadata_path ../../data/OpenVidHD/data_csv/OpenVidHD.csv \
  --degradation_config_path config/degradations.yaml \
  --data_file_keys video,vace_video,vace_reference_image \
  --height 480 --width 832 --num_frames 49 \
  --dataset_repeat 2 \
  --model_id_with_origin_paths Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth \
  --learning_rate 1e-4 --num_epochs 1000000000000000 \
  --remove_prefix_in_ckpt pipe.vace. \
  --output_path /opt/data/private/yyx/data/OpenVidHD/train_pth \
  --trainable_models vace \
  --extra_inputs vace_video,vace_reference_image,degradation_kernels,degradation_params \
  --use_gradient_checkpointing_offload \
  --use_swanlab --swanlab_mode disabled \
  --is_sr --data_process
EOF
)
    ;;

  *)
    echo "❌ Unknown job name: '$SLURM_JOB_NAME'."
    exit 1
    ;;
esac

# 打印并执行命令
echo "=== Launch Command ==="
echo "$CMD"
echo "======================"
eval "$CMD"
