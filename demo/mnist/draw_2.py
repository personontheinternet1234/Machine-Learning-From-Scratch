"""
MNIST model application to user-drawn numbers
"""

import os

import numpy as np
import pygame
from PIL import Image

from gardenpy.models import (
    CNN,
    CNNTorch,
    FNN,
    FNNTorch
)
from gardenpy.utils.data_utils import (
    format_parameters_new,
)
from gardenpy.utils.helper_functions import (
    print_color,
    print_credits
)
from gardenpy.utils.algorithms import (
    Activators
)

model_type = 'fnn'
model_types = ['cnn', 'cnn_torch', 'fnn', 'fnn_torch']
params_root = os.path.join(os.path.dirname(__file__), f'{model_type}_mnist', 'model')
model_base = 'scratch' if not model_type.endswith('_torch') else 'torch'

if model_base == 'scratch':
    weights, biases, kernels = format_parameters_new(params_root, m_type=model_base, )