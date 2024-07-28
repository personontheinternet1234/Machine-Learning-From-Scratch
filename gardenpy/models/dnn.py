r"""
'dnn' includes a Dense Neural Network (DNN) built from GardenPy.

'dnn' includes:
    'DNN': A DNN built from GardenPy.

Refer to 'todo' for in-depth documentation on this model.
"""

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
        self._lyrs = None
        self._g = None
        self._j = None
        self._solver = None
        self._optimizer = None

        # internal parameters
        self._w = None
        self._b = None
        self._zero_grad = None

        # internal dataloaders
        self._train_loader = None
        self._valid_loader = None
        self._batching = None

        # external parameters
        self.thetas = None

        # visual parameters
        self._ansi = ansi_formats()
        self._status = status_bars

    @staticmethod
    def _get_hidden(hidden):
        if hidden is None:
            hidden = [100]
        elif isinstance(hidden, (list, set)):
            for lyr in hidden:
                if not isinstance(lyr, int):
                    raise TypeError('not int')
        return hidden

    @staticmethod
    def _get_thetas(thetas):
        weights = []
        biases = []
        for prm in thetas['weights']:
            if isinstance(prm, Tensor):
                weights.append(prm)
            else:
                weights.append(Initializers(prm['algorithm'], prm['parameters']))
        for prm in thetas['biases']:
            if isinstance(prm, Tensor):
                biases.append(prm)
            else:
                biases.append(Initializers(prm['algorithm'], prm['parameters']))
        return weights, biases

    @staticmethod
    def _get_activators(parameters, lyrs):
        params_norm = {
            'algorithm': 'relu',
            'parameters': None
        }
        params_last = {
            'algorithm': 'softmax',
            'parameters': None
        }
        params = []
        for lyr in range(lyrs - 1):
            params.append(Activators(params_norm['algorithm'], params_norm['parameters']))
        params.append(Activators(params_last['algorithm'], params_last['parameters']))
        return params

    @staticmethod
    def _get_loss(parameters):
        params = {
            'algorithm': 'centropy',
            'parameters': None
        }
        if parameters is not None and isinstance(parameters, dict):
            params.update(parameters)
        elif parameters is not None:
            raise TypeError('not dict')
        return Losses(params['algorithm'], params['parameters'])

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

    def initialize(self, hidden_layers=None, thetas=None, activations=None):
        self._hidden = self._get_hidden(hidden_layers)
        self._w, self._b = self._get_thetas(thetas)
        self._g = self._get_activators(activations)

    def hyperparameters(self, loss=None, optimizer=None):
        self._j = self._get_loss(loss)
        self._solver = self._get_solver()
        self._optimizer = self._get_optimizer(optimizer)

    def forward(self, x: np.ndarray):
        a = [x]
        for lyr in range(len(self._lyrs) - 1):
            a_n = self._g['f'](a[-1] @ self._w[lyr] + self._b[lyr])
            a.append(a_n)
        return a

    def _step(self, a, y):
        grad_w, grad_b = self._solver(a, y)
        self._w = self._optimizer(self._w, grad_w)
        self._b = self._optimizer(self._b, grad_b)

    def _get_solver(self):
        if self._batching == 1:
            def backward(a, y):
                # instantiate gradients
                grad_w = self._zero_grad[0]
                grad_b = self._zero_grad[1]

                # calculate gradients
                grad_a = self._j['d'](y, a[-1])
                for lyr in range(-1, -len(a) + 1, -1):
                    grad_b[lyr] = self._g[lyr]['d'](a[lyr - 1] @ self._w[lyr] + self._b[lyr]) * grad_a
                    grad_w[lyr] = a[lyr - 1].T * grad_b[lyr]
                    grad_a = np.sum(self._w[lyr] * grad_b[lyr], axis=1)
                grad_b[0] = self._g[0]['d'](a[0] @ self._b[0] + self._b[0]) * grad_a
                grad_w[0] = a[0].T * grad_b[0]

                # return gradients
                return grad_w, grad_b
            # return solver
            return backward

        else:
            # redefine zero gradients
            self._zero_grad[0] = np.array([self._zero_grad[0]] * self._batching)
            self._zero_grad[0] = np.array([self._zero_grad[1]] * self._batching)

            def update(a, y):
                # instantiate gradients
                grad_w = self._zero_grad[0]
                grad_b = self._zero_grad[1]

                # calculate gradients
                grad_a = self._j['d'](y, a[-1])
                for lyr in range(-1, -len(a) + 1, -1):
                    grad_b[lyr] = self._g[lyr]['d'](a[lyr - 1] @ self._w[lyr] + self._b[lyr]) * grad_a
                    grad_w[lyr] = np.reshape(a[lyr - 1], (self._batching, self._lyrs[lyr - 1], 1)) * grad_b
                    grad_a = np.reshape(np.sum(self._w[lyr] * grad_b[lyr], axis=2, keepdims=True), (self._batching, 1, self._lyrs[lyr - 1]))
                grad_b[0] = self._g[0]['d'](a[0] @ self._w[0] + self._b[0]) * grad_a
                grad_w[0] = np.reshape(a[0], (self._batching, self._lyrs[0], 1)) * grad_b[0]

                # sum and average gradients
                grad_w = np.sum(grad_w, axis=0) / self._batching
                grad_b = np.sum(grad_b, axis=0) / self._batching

                # return gradients
                return grad_w, grad_b

            # return solver
            return update

    def predict(self, x: np.ndarray):
        return np.argmax(self.forward(x)[-1])

    def fit(self, data, parameters=None):
        b_loss = None
        b_accu = None
        start = time.time()

        print(f"\n{self._ansi['bold']}Training{self._ansi['reset']}")
        for epoch in range(parameters['epochs']):
            x, y = next(data)
            a = self.forward(x)
            self._step(y, a)
            if epoch % parameters['eval_rate'] == 0:
                b_loss = self._j(y, a) / self._batching
                b_accu = 0.5 * np.sum(np.abs(a - y)) / self._batching
            if self._status:
                print("Training")
                desc = (
                    f"{str(epoch + 1).zfill(len(str(parameters['max_iter'])))}{self._ansi['white']}it{self._ansi['reset']}/{parameters['max_iter']}{self._ansi['white']}it{self._ansi['reset']}  "
                    f"{(100 * (epoch + 1) / parameters['max_iter']):05.1f}{self._ansi['white']}%{self._ansi['reset']}  "
                    f"{b_loss:05}{self._ansi['white']}loss{self._ansi['reset']}  "
                    f"{b_accu:05.1f}{self._ansi['white']}accu{self._ansi['reset']}  "
                    f"{convert_time(time.time() - start)}{self._ansi['white']}et{self._ansi['reset']}  "
                    f"{convert_time((time.time() - start) * parameters['max_iter'] / (epoch + 1) - (time.time() - start))}{self._ansi['white']}eta{self._ansi['reset']}  "
                    f"{round((epoch + 1) / (time.time() - start), 1)}{self._ansi['white']}it/s{self._ansi['reset']}"
                )
                progress(epoch, parameters['max_iter'], desc=desc)

            print(time.time() - start)
