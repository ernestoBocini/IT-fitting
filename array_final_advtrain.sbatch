#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 100G
#SBATCH --time 12:00:00

source ~/.bashrc

conda activate it-fitting
cd it-fitting

nvidia-smi


seed=0
date=220505
arch=cornet_s
loss=logCKA
labels=1


# Imagenet (control)
if [ $SLURM_ARRAY_TASK_ID == 0 ]; 
then
    mix=1
    sp=${date}-final-AT2-labels_${labels}-mix_${mix}
    CUDA_VISIBLE_DEVICES=0 python main.py -v --seed $seed --neural_loss $loss --arch $arch --epochs 60 --save_path $sp -nd sachimajajhongpublic -s All -n All \
        -aei \
        --loss_weights 1 0 1 -mix_rate $mix -causal 1 --val_every 30 -ati -teps 0.015686
fi

## Imagenet (control)
#if [ $SLURM_ARRAY_TASK_ID == 0 ]; 
#then
#    mix=1
#    sp=${date}-final-AT-labels_${labels}-mix_${mix}
#    python main.py -v --seed $seed --neural_loss $loss --arch $arch --epochs 1200 --save_path $sp -nd sachimajajhongpublic -s All -n All \
#        -BS dicarlo.Rajalingham2018-i2n dicarlo.Kar2022human-i2n dicarlo.Kar2018-i2n dicarlo.Rajalingham2018subset-i2n -aei \
#        --loss_weights 1 0 1 -mix_rate $mix -causal 1 --val_every 30 -ati -teps 0.015686
#fi
#
# Imagenet (control)
#if [ $SLURM_ARRAY_TASK_ID == 1 ]; 
#then
#    mix=1
#    sp=${date}-final-ATneural-labels_${labels}-mix_${mix}
#    python main.py -v --seed $seed --neural_loss $loss --arch $arch --epochs 1200 --save_path $sp -nd sachimajajhongpublic -s All -n All \
#        -BS dicarlo.Rajalingham2018-i2n dicarlo.Kar2022human-i2n dicarlo.Kar2018-i2n dicarlo.Rajalingham2018subset-i2n -aei \
#        --loss_weights 1 1 1 -mix_rate $mix -causal 1 --val_every 30 -ati -teps 0.015686
#fi
