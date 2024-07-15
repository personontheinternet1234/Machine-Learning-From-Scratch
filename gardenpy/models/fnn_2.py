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
        self._initializer = None
        self._activator = None
        self._loss = None
        self._optimizer = None

        # trainable parameters
        self.thetas = None

        # calculation variables
        self._zero_grad = None

        # visual parameters
        self._status = not status_bars
        # todo: check out how FORE coloring works
        self._color = f'{Fore.GREEN}{{l_bar}}{{bar}}{{r_bar}}{Style.RESET_ALL}'

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
                'parameters': 'default'
            },
            'biases': {
                'algorithm': 'zeros',
                'parameters': 'default'
            }
        }
        default_activators = ['relu' for lyr in default_layers].append('softmax')

        layers = layer_sizes  # todo
        activators = ...

        self._layers = layers
        self._initializers = initializers
        self._activators = activators

    def initialize_solver(self, loss=None, optimizer=None):
        ...
