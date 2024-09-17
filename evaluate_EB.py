import os, glob, argparse, random

import sys
import os
import functools
import torch
import json
import sys
import os
from os.path import join as pjoin
import argparse


import pandas as pd
import numpy as np
import torch as ch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from candidate_models.base_models.cornet import TemporalPytorchWrapper
from model_tools.activations.pytorch import PytorchWrapper, load_preprocess_images
from model_tools.brain_transformation import ModelCommitment

from brainscore import score_model as _score_model
from candidate_models.model_commitments.cornets import (
    CORnetCommitment,
    CORNET_S_TIMEMAPPING,
    _build_time_mappings,
)
from cornet import cornet_s

from datamodules.neural_datamodule import NeuralDataModule
from models.helpers import layer_maps, add_normalization, add_outputs, Mask

def load_model(path_to_weights):
    # load the model architecture with the normalization preprocessor
    model = cornet_s(pretrained=False)
    model = add_normalization(model, normalization=layer_maps['cornet_s']['normalization'])
    model = add_outputs(model, out_name='decoder.linear', n_outputs=8)

    # load weights and strip pesky 'model.' prefix
    state_dict = ch.load(path_to_weights)
    weights = {k.replace('model.', '') :v for k,v in state_dict['state_dict'].items()}

    # load the architecture with the trained model weights
    model.load_state_dict(weights)

    # model in eval mode............
    model.eval()

    return model


def wrap_model(model, identifier, image_size=224):
    image_size = 224
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size, normalize_mean=(0,0,0), normalize_std=(1,1,1))
    wrapper = PytorchWrapper(identifier=identifier, model=model, preprocessing=preprocessing)
    wrapper.image_size = image_size
    return wrapper



def score_model(
    model_identifier, brain_model, benchmark_identifier, layers=[], image_size=224
):
    os.environ["RESULTCACHING_DISABLE"] = "brainscore.score_model,model_tools"
    

    score = _score_model(
        model_identifier=model_identifier,
        model=brain_model,
        benchmark_identifier=benchmark_identifier,
    )

    return score


path_to_weights = '/work/upschrimpf1/bocini/domain-transfer/IT-fitting/logs/240201-final-labels_0-mix_1/model_cornet_s-loss_logCKA-ds_sachimajajhongpublic-fanimals_All-neurons_All-stimuli_All-seed_5/version_1/checkpoints/epoch=1199-step=26399.ckpt'
identifier = 'btCORnet_S'
    
    
model = load_model(path_to_weights)
wrapped_model = wrap_model(model, identifier)
layers = ['1.module.'+layer for layer in ['V1', 'V2', 'V4', 'IT', 'decoder.avgpool']]
brain_model = ModelCommitment(identifier=identifier, activations_model=wrapped_model, 
        layers=layers)

benchmarks = [
    'Igustibagus2024.IT_readout-accuracy'
]

def update_json_file(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


scores = {}
for benchmark_identifier in tqdm(benchmarks):
    print(f">>>>>> Start Benchmark {benchmark_identifier}")
    score = score_model(identifier, brain_model, benchmark_identifier)
    scores[benchmark_identifier] = score.values[0]
    print(f">>>>> Score = {score}")


update_json_file(f'./benchmark.json', scores)
print("Evaluation of all models completed.")