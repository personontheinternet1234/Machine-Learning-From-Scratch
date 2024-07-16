"""
todo: write description
"""

import random
import time
import warnings

import numpy as np
import pandas as pd

from ..utils import (
    Initializers,
    Activators,
    Losses,
    Optimizers,
)


class FNN:
    def __init__(self, status_bars: bool = True):
        # hyperparameters
        self._hidden = None
        self._activators = None
        self._loss = None
        self._optimizer = None
        self._batching = None

        # trainable parameters
        self.thetas = None

        # non trainable parameters
        self.x = None
        self.y = None

        # calculation variables
        self._zero_grad = None

        # visual parameters
        self._status = status_bars

    @staticmethod
    def _get_hidden(hidden):
        # todo: comment and fix errors
        if hidden is None:
            hidden = [100]
        elif isinstance(hidden, (list, np.ndarray)):
            for lyr in hidden:
                if not isinstance(lyr, (int, np.int64)):
                    raise TypeError('stop')
        return hidden

    def _get_thetas(self, thetas):
        ...

    def _get_activations(self):
        ...

    @staticmethod
    def _get_initializer(algorithm, parameters):
        return Initializers(algorithm, parameters)

    @staticmethod
    def _get_activator(algorithm, parameters):
        return Activators(algorithm, parameters)

    @staticmethod
    def _get_loss(parameters):
        default = {
            'algorithm': 'centropy',
            'parameters': None
        }
        if not isinstance(parameters, dict):
            raise TypeError('stop')
        for prm in parameters:
            if prm not in default:
                warnings.warn('warning')
            else:
                default[prm] = parameters[prm]
        return Optimizers(default['algorithm'], default['parameters'])

    @staticmethod
    def _get_optimizer(parameters):
        default = {
            'algorithm': 'adam',
            'hyperparameters': None
        }
        if not isinstance(parameters, dict):
            raise TypeError('stop')
        for prm in parameters:
            if prm not in default:
                warnings.warn('warning')
            else:
                default[prm] = parameters[prm]
        return Optimizers(default['algorithm'], default['hyperparameters'])

    def _get_batching(self, batching):
        ...

    def instantiate(self, hidden_layers=None, thetas=None, activations=None):
        self._hidden = self._get_hidden(hidden_layers)

    def hyperparameters(self, loss=None, optimizer=None):
        self._loss = self._get_loss(loss)
        self._optimizer = self._get_optimizer(optimizer)

    def forward(self):
        ...

    def fit(self, x, y, batch_size=None, max_iter=100000):
        if not isinstance((x, y), np.ndarray):
            raise TypeError('numpy')
        self.x = x
        self.y = y
        self._batching = self._get_batching(batch_size)

        start = time.time()
        for iter in max_iter:

