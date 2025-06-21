#!/bin/bash
#SBATCH --job-name=wan_vace_test
#SBATCH --output=logs/wan_vace_test_%j.out
#SBATCH --error=logs/wan_vace_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ai

# 激活 conda 环境
/HOME/paratera_xy/pxy545/miniconda3/bin/conda init             # 确保 conda 命令可用（必要）
conda activate diffsynth           # 替换为你的实际 conda 环境名

# （可选）显示当前环境和 CUDA 信息，方便调试
echo "Conda env: $CONDA_DEFAULT_ENV"
which python
nvidia-smi

# 训练命令
torchrun --standalone --nproc_per_node=1 examples/wanvideo/model_inference/test_batch.py --checkpoint exp/train/Wan2.1-VACE-1.3B_full_4gpu_datapth