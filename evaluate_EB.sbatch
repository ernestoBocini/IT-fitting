#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 50G
#SBATCH --time 4:00:00

source ~/.bashrc

conda activate it-fitting-bs
cd it-fitting-bs

nvidia-smi

python evaluate_EB.py