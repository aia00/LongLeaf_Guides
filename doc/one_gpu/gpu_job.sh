#! /bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p a100-gpu
#SBATCH --mem=3g
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1

module purge

module load python/3.8.8
module load cuda

source myenv/bin/activate

python finetune_fp4_opt_bnb_peft.py
