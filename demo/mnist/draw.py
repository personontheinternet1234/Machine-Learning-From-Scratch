r"""
Trained Model application for MNIST on user-drawn numbers.
"""

import os

import numpy as np
import pygame
from PIL import Image

from gardenpy import (
    CNN,
    CNNTorch,
    DNN,
    DNNTorch
)
from gardenpy.utils.algorithms import Activators

# model type
model_type = 'dnn'
model_types = ['cnn', 'cnn_torch', 'dnn', 'dnn_torch']


def draw():
    parameters_root = os.path.join(os.path.dirname(__file__), f'{model_type}_mnist', 'model')
    model_base = 'scratch' if not model_type.endswith('_torch') else 'torch'

    if model_base == 'scratch':
        ...


if __name__ == '__main__':
    draw()
