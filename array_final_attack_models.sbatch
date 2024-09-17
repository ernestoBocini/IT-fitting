#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 100000M
#SBATCH --time 40:00:00
#SBATCH -o slurm_logs/score_models_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e slurm_logs/score_models_%A_%a.err  # File to which STDERR will be written, %j inserts jobid

### for multiple GPUS #SBATCH --array=0-2

source ~/.bashrc

conda activate it-fitting-bs

nvidia-smi

seed=3
date=220505
arch=cornet_s
loss=logCKA
labels=1

nvidia-smi
#sp='new_final_models'
sp='new_AT2_final_models'
#ds='COCO'
ds='HVM_var6'

#norm='inf'
#
#epss=(
#   0.00003063725490196
#   0.00006127450980392 
#   0.00012254901960784 
#   0.00024509803921568 
#   0.00049019607843137 
#   0.00098039215686274 
#   0.00196078431372549 
#   0.00392156862745098 
#   0.00784313725490196 
#   0.01568627450980392 
#)

norm=2

epss=(
    0.00390625
    0.0078125
    0.015625
    0.03125
    0.0625
    0.125
    0.25
    0.5
    1
    2
)

# HVM labels
for eps in ${epss[${SLURM_ARRAY_TASK_ID}]}
do
    python evaluate_model.py --seed 6 --dataset=${ds} --logdir=$sp --norm=$norm --eps=$eps
done
