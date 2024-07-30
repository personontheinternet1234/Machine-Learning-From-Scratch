r"""
'dnn' includes a Dense Neural Network (DNN) built from GardenPy.

'dnn' includes:
    'DNN': A DNN built from GardenPy.

Refer to 'todo' for in-depth documentation on this model.
"""

import time
from typing import Union
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
    ansi
)


class DNN:
    def __init__(self, status_bars: bool = False):
        # hyperparameters
        self._hidden = None
        self._lyrs = None
        self._g = None
        self._j = None
        self._optimizer = None

        # internal parameters
        self._w = None
        self._b = None
        self._a = None
        self._alpha = None
        self._beta = None
        self._loss = None
        self._grad_w = None
        self._grad_b = None
        self._zeros = None
        self._train_loader = None

        # visual parameters
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
    def _get_thetas(parameters, hdns):
        # set default initializers
        weights = [{'algorithm': 'gaussian'}] * (hdns + 1)
        biases = [{'algorithm': 'gaussian'}] * (hdns + 1)
        if parameters is not None:
            # defined parameters
            for lyr in range(hdns + 1):
                try:
                    # valid index
                    if isinstance(parameters['weights'][lyr], (Tensor, dict)):
                        # valid parameter
                        weights[lyr] = parameters['weights'][lyr]
                    else:
                        # invalid datatype for parameter
                        raise TypeError(
                            f"Invalid datatype for 'thetas': '{}'\n"
                            f"Choose from: '{dtypes[self._algorithm][prm]}'"
                        )
                except IndexError:
                    # invalid index
                    warnings.warn('no instance')
                    pass

        for w in range(len(weights)):
            if isinstance(weights, dict) and 'algorithm' in weights[w]:
                alg = weights[w]['algorithm']
                del weights[w]['algorithm']
                weights[w] = Initializers(alg, weights[w])
            else:
                raise ValueError('no algorithm')
        return weights, biases
        # for prm in thetas['weights']:
        #     if isinstance(prm, Tensor):
        #         weights[]
        #     else:
        #         if 'algorithm' not in prm:
        #             raise ValueError('no specified algorithm')
        #         alg = prm['algorithm']
        #         del prm['algorithm']
        #         weights.append(Initializers(alg, prm))
        # for prm in thetas['biases']:
        #     if isinstance(prm, Tensor):
        #         biases.append(prm)
        #     else:
        #         if 'algorithm' not in prm:
        #             raise ValueError('no specified algorithm')
        #         alg = prm['algorithm']
        #         del prm['algorithm']
        #         biases.append(Initializers(alg, prm))
        # return weights, biases

    @staticmethod
    def _get_activators(parameters, hdns):
        params_norm = {
            'algorithm': 'relu',
            'parameters': None
        }
        params_last = {
            'algorithm': 'softmax',
            'parameters': None
        }
        params = []
        for hdn in range(hdns - 1):
            if parameters is None:
                params.append(Activators(params_norm['algorithm'], params_norm['parameters']))
            elif isinstance(parameters[hdn], dict):
                params.append(Activators(parameters[hdn]['algorithm'], parameters[hdn]['parameters']))
            elif isinstance(parameters[hdn], np.ndarray):
                params.append(Tensor(parameters[hdn]))
            elif isinstance(parameters[hdn], Tensor):
                params.append(parameters[hdn])
            else:
                raise TypeError('not dict')
        if parameters is None:
            params.append(Activators(params_last['algorithm'], params_last['parameters']))
        elif isinstance(parameters[hdns], dict):
            params.append(Activators(parameters[hdns]['algorithm'], parameters[hdns]['parameters']))
        elif isinstance(parameters[hdns], np.ndarray):
            params.append(Tensor(parameters[hdns]))
        elif isinstance(parameters[hdns], Tensor):
            params.append(parameters[hdns])
        else:
            raise TypeError('not dict')
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

    def initialize(self, hidden_layers=None, thetas=None, activations=None):
        self._hidden = self._get_hidden(hidden_layers)
        self._w, self._b = self._get_thetas(thetas)
        self._g = self._get_activators(activations, self._hidden)

    def hyperparameters(self, loss=None, optimizer=None):
        self._j = self._get_loss(loss)
        self._optimizer = self._get_optimizer(optimizer)

    def forward(self, x: Union[Tensor, np.ndarray]) -> np.ndarray:
        if not isinstance(x, (Tensor, np.ndarray)):
            raise TypeError('not correct type')
        if isinstance(x, Tensor):
            x = x.to_array()
        a = self._zeros
        a[0] = x
        for lyr in range(len(self._lyrs) - 1):
            a[lyr + 1] = self._g[lyr](a[lyr] @ self._w[lyr].to_array() + self._b[lyr].to_array())
        return a

    def _forward(self, x, y):
        self._a = self._zeros
        self._a[0] = x
        self._alpha = self._zeros
        self._beta = self._zeros
        for lyr in range(len(self._lyrs) - 1):
            self._alpha[lyr] = self._a[-1] @ self._w[lyr]
            self._beta[lyr] = self._alpha[-1] + self._b[lyr]
            self._a[lyr + 1] = self._g[lyr](self._beta[-1])
        self._loss = self._j(self._a[-1], y)

    def _backward(self):
        grad_alpha = nabla(self._loss, self._alpha[-1])
        self._grad_b = self._zeros
        self._grad_w = self._zeros
        for lyr in range(-1, -self._lyrs + 1, -1):
            self._grad_b[lyr] = chain(grad_alpha, nabla(self._alpha[lyr], self._b[lyr]))
            self._grad_w[lyr] = chain(self._grad_b[lyr], nabla(self._b[lyr], self._w[lyr]))
            grad_alpha = chain(grad_alpha, nabla(self._alpha[lyr - 1], self._grad_w[lyr]))
        self._grad_b[0] = chain(grad_alpha, nabla(self._alpha[1], self._b[0]))
        self._grad_w[0] = chain(self._grad_b[0], nabla(self._b[0], self._w[0]))
        if self._train_loader.batching > 1:
            self._grad_b = Tensor(np.sum(self._grad_b.to_array(), axis=0))
            self._grad_w = Tensor(np.sum(self._grad_w.to_array(), axis=0))

    def _step(self, x, y):
        self._forward(x, y)
        self._backward()
        self._w = self._optimizer(self._w, self._grad_w)
        self._b = self._optimizer(self._b, self._grad_b)

    def output(self, x: Union[Tensor, np.ndarray]) -> np.ndarray:
        return self.forward(x)[-1]

    def predict(self, x: Union[Tensor, np.ndarray]) -> np.int64:
        return np.argmax(self.forward(x)[-1])

    def fit(self, data, parameters=None):
        b_loss = None
        b_accu = None
        start = time.time()

        print(f"\n{ansi['bold']}Training{ansi['reset']}")
        for epoch in range(parameters['epochs']):
            x, y = next(data)
            self._step(x, y)
            if epoch % parameters['eval_rate'] == 0:
                b_loss = self._loss / self._train_loader.batching
                b_accu = 0.5 * np.sum(np.abs(self._a[-1] - y)) / self._train_loader.batching
            if self._status:
                print("Training")
                desc = (
                    f"{str(epoch + 1).zfill(len(str(parameters['max_iter'])))}{ansi['white']}it{ansi['reset']}/{parameters['max_iter']}{ansi['white']}it{ansi['reset']}  "
                    f"{(100 * (epoch + 1) / parameters['max_iter']):05.1f}{ansi['white']}%{ansi['reset']}  "
                    f"{b_loss:05}{ansi['white']}loss{ansi['reset']}  "
                    f"{b_accu:05.1f}{ansi['white']}accu{ansi['reset']}  "
                    f"{convert_time(time.time() - start)}{ansi['white']}et{ansi['reset']}  "
                    f"{convert_time((time.time() - start) * parameters['max_iter'] / (epoch + 1) - (time.time() - start))}{ansi['white']}eta{ansi['reset']}  "
                    f"{round((epoch + 1) / (time.time() - start), 1)}{ansi['white']}it/s{ansi['reset']}"
                )
                progress(epoch, parameters['max_iter'], desc=desc)

            print(time.time() - start)

    def final(self):
        return self._w, self._b
