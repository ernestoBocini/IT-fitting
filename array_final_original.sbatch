#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 100G
#SBATCH --time 30:00:00


### for multiple GPUS #SBATCH --array=0-2

source ~/.bashrc

conda activate it-fitting-bs
cd it-fitting

nvidia-smi

seed=5
date=240201
arch=cornet_s
loss=logCKA
labels=0

# Imagenet (control)
mix=1
sp=${date}-final-labels_${labels}-mix_${mix}
python main.py -v --seed $seed --neural_loss $loss --arch $arch --epochs 1200 --save_path $sp -nd sachimajajhongpublic -s All -n All \
    -BS dicarlo.Rajalingham2018public-i2n Igustibagus2024.IT_readout-accuracy -aei \
    --loss_weights 1 1 ${labels} -mix_rate $mix -causal 1 --val_every 30
