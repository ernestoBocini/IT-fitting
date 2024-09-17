#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 100G
#SBATCH --time 12:00:00

source ~/.bashrc

conda activate it-fitting
cd it-fitting

nvidia-smi

seed=5
date=220505
arch=cornet_s
loss=logCKA
labels=0

# Imagenet (control)
if [ $SLURM_ARRAY_TASK_ID == 0 ]; 
then
    mix=1
    sp=${date}-final-labels_${labels}-mix_${mix}
    CUDA_VISIBLE_DEVICES=0 python main.py -v --seed $seed --neural_loss $loss --arch $arch --epochs 1200 --save_path $sp -nd sachimajajhongpublic -s All -n All \
        -aei \
        --loss_weights 1 1 ${labels} -mix_rate $mix -causal 1 --val_every 30
fi

if [ $SLURM_ARRAY_TASK_ID == 1 ]; 
then
    mix=0.5
    sp=${date}-final-labels_${labels}-mix_${mix}
    CUDA_VISIBLE_DEVICES=0 python main.py -v --seed $seed --neural_loss $loss --arch $arch --epochs 1200 --save_path $sp -nd sachimajajhongpublic -s All -n All \
        -aei \
        --loss_weights 1 1 ${labels} -mix_rate $mix -causal 1 --val_every 30
fi

if [ $SLURM_ARRAY_TASK_ID == 2 ]; 
then
    mix=0.25
    sp=${date}-final-labels_${labels}-mix_${mix}
    CUDA_VISIBLE_DEVICES=0 python main.py -v --seed $seed --neural_loss $loss --arch $arch --epochs 1200 --save_path $sp -nd sachimajajhongpublic -s All -n All \
        -aei \
        --loss_weights 1 1 ${labels} -mix_rate $mix -causal 1 --val_every 30
fi

if [ $SLURM_ARRAY_TASK_ID == 3 ]; 
then
    mix=0.125
    sp=${date}-final-labels_${labels}-mix_${mix}
    CUDA_VISIBLE_DEVICES=0 python main.py -v --seed $seed --neural_loss $loss --arch $arch --epochs 1200 --save_path $sp -nd sachimajajhongpublic -s All -n All \
        -aei \
        --loss_weights 1 1 ${labels} -mix_rate $mix -causal 1 --val_every 30
fi

if [ $SLURM_ARRAY_TASK_ID == 4 ]; 
then
    mix=0.0625
    sp=${date}-final-labels_${labels}-mix_${mix}
    CUDA_VISIBLE_DEVICES=0 python main.py -v --seed $seed --neural_loss $loss --arch $arch --epochs 1200 --save_path $sp -nd sachimajajhongpublic -s All -n All \
        -aei \
        --loss_weights 1 1 ${labels} -mix_rate $mix -causal 1 --val_every 30
fi

if [ $SLURM_ARRAY_TASK_ID == 5 ]; 
then
    mix=0.03125
    sp=${date}-final-labels_${labels}-mix_${mix}
    CUDA_VISIBLE_DEVICES=0 python main.py -v --seed $seed --neural_loss $loss --arch $arch --epochs 1200 --save_path $sp -nd sachimajajhongpublic -s All -n All \
        -aei \
        --loss_weights 1 1 ${labels} -mix_rate $mix -causal 1 --val_every 30
fi

if [ $SLURM_ARRAY_TASK_ID == 6 ]; 
then
    mix=0
    sp=${date}-final-labels_${labels}-mix_${mix}
    CUDA_VISIBLE_DEVICES=0 python main.py -v --seed $seed --neural_loss $loss --arch $arch --epochs 1200 --save_path $sp -nd sachimajajhongpublic -s All -n All \
        -aei \
        --loss_weights 1 1 ${labels} -mix_rate $mix -causal 1 --val_every 30
fi
