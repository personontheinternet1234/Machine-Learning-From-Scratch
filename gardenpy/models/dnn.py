r"""
'dnn' includes a Dense Neural Network (DNN) built from GardenPy.

'dnn' includes:
    'DNN': A DNN built from GardenPy.

Refer to 'todo' for in-depth documentation on this model.
"""

import random
import time
import warnings

import numpy as np

from ..utils.operators import (
    nabla,
    chain
)

from ..utils.objects import Tensor

from ..utils.algorithms import (
    Initializers,
    Activators,
    Losses,
    Optimizers
)

from ..utils.helper_functions import (
    progress,
    convert_time,
    ansi_formats
)


class DNN:
    def __init__(self, status_bars: bool = False):
        # hyperparameters
        self._hidden = None
        self._activators = None
        self._loss = None
        self._optimizer = None
        self._batching = None

        # internal parameters
        self._thetas = None
        self._zero_grad = None

        # external parameters
        self.thetas = None  # todo: eventually a dictionary

        # visual parameters
        self._ansi = ansi_formats()
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
    def _get_activators(algorithm, parameters):
        return Activators(algorithm, parameters)

    @staticmethod
    def _get_loss(parameters):
        default = {
            'algorithm': 'centropy',
            'parameters': None
        }
        if parameters is None:
            parameters = default
        if not isinstance(parameters, dict):
            raise TypeError('stop')
        for prm in parameters:
            if prm not in default:
                warnings.warn('warning')
        for prm in default:
            if prm not in parameters:
                raise ValueError('stop')
        return Losses(parameters['algorithm'], parameters['parameters'])

    @staticmethod
    def _get_optimizer(parameters):
        default = {
            'algorithm': 'adam',
            'hyperparameters': None
        }
        if parameters is None:
            parameters = default
        if not isinstance(parameters, dict):
            raise TypeError('stop')
        for prm in parameters:
            if prm not in default:
                warnings.warn('warning')
        for prm in default:
            if prm not in parameters:
                raise ValueError('stop')
        return Optimizers(parameters['algorithm'], parameters['hyperparameters'])

    def _get_batching(self, batching):
        ...

    def _step(self, x, y):
        ...

    def initialize(self, hidden_layers=None, thetas=None, activations=None):
        self._hidden = self._get_hidden(hidden_layers)
        self._thetas = self._get_thetas(thetas)
        self._activators = self._get_activators(activations)

    def hyperparameters(self, loss=None, optimizer=None):
        self._loss = self._get_loss(loss)
        self._optimizer = self._get_optimizer(optimizer)

    def forward(self, x: np.ndarray):
        ...

    def predict(self, x: np.ndarray):
        return np.argmax(self.forward(x)[-1])

    def validation(self, valid_x: np.ndarray, valid_y: np.ndarray, parameters: (dict, None) = None):
        warnings.warn(f"Not supported yet.")

    def fit(self, x: np.ndarray, y: np.ndarray, parameters=None):
        if not isinstance((x, y), np.ndarray):
            raise TypeError('numpy')
        self.x = x
        self.y = y
        self._xy_len = len(y)
        self._batching = self._get_batching(...)

        start = time.time()
        for i in range(parameters['max_iter']):
            tc = random.randint(self._batching, self._xy_len)
            x, y = x[tc-self._batching:tc], y[tc-self._batching:tc]

            yhat = self.forward(x)
            self._step(y, yhat)
            loss = None
            accu = None
            if i % ... == 0:
                loss = self._loss(y, yhat) / self._batching
                accu = 0.5 * np.sum(np.abs(yhat - y)) / self._batching

            if self._status:
                print("Training")
                desc = (
                    f"{str(i + 1).zfill(len(str(parameters['max_iter'])))}{self._ansi['white']}it{self._ansi['reset']}/{parameters['max_iter']}{self._ansi['white']}it{self._ansi['reset']}  "
                    f"{(100 * (i + 1) / parameters['max_iter']):05.1f}{self._ansi['white']}%{self._ansi['reset']}  "
                    f"{loss:05}{self._ansi['white']}loss{self._ansi['reset']}  "
                    f"{accu:05.1f}{self._ansi['white']}accu{self._ansi['reset']}  "
                    f"{convert_time(time.time() - start)}{self._ansi['white']}et{self._ansi['reset']}  "
                    f"{convert_time((time.time() - start) * parameters['max_iter'] / (i + 1) - (time.time() - start))}{self._ansi['white']}eta{self._ansi['reset']}  "
                    f"{round((i + 1) / (time.time() - start), 1)}{self._ansi['white']}it/s{self._ansi['reset']}"
                )
                progress(i, parameters['max_iter'], desc=desc)

            print(time.time() - start)
