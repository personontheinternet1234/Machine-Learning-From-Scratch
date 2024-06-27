"""
MNIST model application to user-drawn numbers
"""

import os

import numpy as np
import pygame
from PIL import Image

from garden.models.fnn import (
    FNN
)
from garden.utils.data_utils import (
    format_parameters,
)
from garden.utils.helper_functions import (
    print_color,
    print_credits
)
from garden.utils.functional import (
    softmax
)

model_type = 'fnn'
params_root = os.path.join(os.path.dirname(__file__), f'{model_type}_mnist', 'model')

