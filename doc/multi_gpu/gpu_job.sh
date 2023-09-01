#!/bin/bash
#SBATCH --qos gpu_access
#SBATCH --partition=a100-gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2

module purge
module load python/3.8.8
module load cuda

source my_env/bin/activate

export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=12345  # Choose any available port


WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun \
--nproc_per_node=2 \
--master_port=12345 \
finetune.py \
--base_model 'decapoda-research/llama-7b-hf' \
--data_path 'trans_chinese_alpaca_data.json' \
--output_dir './lora-alpaca-zh'
