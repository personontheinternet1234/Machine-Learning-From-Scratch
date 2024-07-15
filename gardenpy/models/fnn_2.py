"""
Since I've updated so many things, I basically need to rewrite the FNN class
todo: write description
"""

import random
import time

import numpy as np
import pandas as pd

from gardenpy.utils.algorithms import (
    Initializers,
    Activators,
    Losses,
    Optimizers
)
from gardenpy.metrics.metrics import (
    cm
)
from gardenpy.utils.data_utils import (
    shuffle
)

from colorama import Fore, Style
from tqdm import tqdm


class FNN:
    def __init__(self, status_bars=True):
        # hyperparameters
        self._layers = None
        self._initializers = None
        self._activators = None
        self._loss = None
        self._optimizer = None

        # trainable parameters
        self.thetas = None

        # calculation variables
        self._zero_grad = None

        # visual parameters
        self._status = not status_bars

    @staticmethod
    def _get_initializer(algorithm, parameters):
        return Initializers(algorithm, parameters)

    @staticmethod
    def _get_activator(algorithm, parameters):
        return Activators(algorithm, parameters)

    @staticmethod
    def _get_loss(algorithm, parameters):
        return Losses(algorithm, parameters)

    @staticmethod
    def _get_optimizer(algorithm, hyperparameters):
        return Optimizers(algorithm, hyperparameters)

    def initialize_model(self, layer_sizes=None, initialization_methods=None, activation_methods=None):
        default_layers = ['auto', 100, 'auto']
        default_initializers = {
            'weights': {
                'algorithm': 'xavier',
                'parameters': None
            },
            'biases': {
                'algorithm': 'zeros',
                'parameters': None
            }
        }
        if initialization_methods:
            # todo: add checks
            init_mts = initialization_methods
        else:
            init_mts = default_initializers

        if not layer_sizes:
            self._layers = default_layers
        else:
            if not isinstance(layer_sizes, (np.ndarray, list)):
                raise ValueError(f"'layer_sizes' is not a list or numpy array: {layer_sizes}")
            elif len(layer_sizes) <= 2:
                raise ValueError(
                    f"Invalid length for 'layer_sizes: {layer_sizes}"
                    f"'layer_sizes': must be greater than 1"
                )
            for lyr in layer_sizes:
                if not isinstance(lyr, (int, np.int64)) or lyr == 'auto':
                    raise ValueError(f"'layer' is not an integer or 'auto': {lyr}")
            self._layers = layer_sizes
        self._initializers = {
            'weights': self._get_initializer(
                init_mts['weights']['algorithm'],
                init_mts['weights']['parameters']
            ),
            'biases': self._get_initializer(
                init_mts['biases']['algorithm'],
                init_mts['biases']['parameters']
            )
        }
        if not activation_methods:
            ...

    def initialize_solver(self, loss=None, optimizer=None):
        default_loss = {
            'algorithm': 'centropy',
            'parameters': 'default'
        }
        default_optimizer = {
            'algorithm': 'adam',
            'hyperparameters': None
        }

