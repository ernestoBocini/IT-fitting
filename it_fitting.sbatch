#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 100G
#SBATCH --time 12:00:00

source ~/.bashrc

conda activate it-fitting-bs
cd it-fitting-bs

nvidia-smi

seed=5
date=220505
arch=cornet_s
loss=logCKA
labels=0

source ~/.bashrc

conda activate it-fitting-bs
cd it-fitting

nvidia-smi

CUDA_VISIBLE_DEVICES=0 python main.py -v --seed 0 --neural_loss logCKA --arch cornet_s --epochs 1000 --save_path EBnew -nd sachimajajhongpublic -s All -n All \
 -BS Igustibagus2024.IT_readout-accuracy -aei \
 --loss_weights 1 1 1 -mix_rate 1 -causal 1 --val_every 30