"""
mathematical functions
"""

import numpy as np
from typing import Union


def initializers(name, gain='auto'):
    if gain == 'auto':
        gains = {
            'xavier': 1
        }
        if name in gains:
            gain = gains[name]
        else:
            gain = None
    init = {
        'gaussian': lambda l, w: np.random.randn(l, w),
        'xavier': lambda l, w: np.random.randn(l, w) * gain / np.sqrt(2 / (w + l)),
        'zeros': lambda l, w: np.zeros((l, w)),
        'ones': lambda l, w: np.ones((l, w))
    }
    if name in init:
        return init[name]
    else:
        raise ValueError(f"'{name}' is an invalid initialization method")


def activators(name, beta='auto'):
    """ get raw activation function """
    # set beta
    if beta == 'auto':
        betas = {
            'leaky relu': 0.01,
            'softplus': 1,
            'mish': 1
        }
        if name in betas:
            beta = betas[name]
        else:
            beta = None
    elif isinstance(beta, float):
        beta = beta
    else:
        raise ValueError(f"'beta' ({beta}) must be a float or 'auto'")
    # set activation function
    g = {
        'softmax': lambda x: np.exp(x) / np.sum(np.exp(x)),
        'relu': lambda x: np.maximum(0, x),
        'leaky-relu': lambda x: np.maximum(beta * x, x),
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'tanh': lambda x: np.tanh(x),
        'softplus': lambda x: np.log(1 + np.exp(beta * x)) / beta,
        'mish': lambda x: x * np.tanh(np.log(1 + np.exp(beta * x)) / beta)
    }
    # return activation function
    if name in g:
        return g[name]
    else:
        raise ValueError(f"'{name}' is an invalid activation function")


def d_activators(name, beta='auto'):
    """ get derivative of raw activation function """
    # set beta
    if beta == 'auto':
        betas = {
            'leaky relu': 0.01,
            'softplus': 1,
            'mish': 1
        }
        if name in betas:
            beta = betas[name]
        else:
            beta = 0
    elif isinstance(beta, float):
        beta = beta
    else:
        raise ValueError(f"'beta' ({beta}) must be a float or 'auto'")
    dg = {
        'softmax': lambda x: ...,  # todo: write this activation algorithm
        'relu': lambda x: np.where(x > 0, 1, 0),
        'leaky-relu': lambda x: np.where(x > 0, 1, beta),
        'sigmoid': lambda x: np.exp(-x) / ((1 + np.exp(-x)) ** 2),
        'tanh': lambda x: np.cosh(x) ** -2,
        'softplus': lambda x: beta * np.exp(beta * x) / (beta + beta * np.exp(beta * x)),
        'mish': lambda x: (np.tanh(np.log(1 + np.exp(beta * x)) / beta)) + (x * (np.cosh(np.log(1 + np.exp(beta * x)) / beta) ** -2) * (beta * np.exp(beta * x) / (beta + beta * np.exp(beta * x))))
    }
    if name in dg:
        return dg[name]
    else:
        raise ValueError(f"'{name}' is an invalid activation function")


def losses(name):
    """ get raw cost function """
    j = {
        'ssr': lambda y, yhat: np.sum((y - yhat) ** 2),
        'sar': lambda y, yhat: np.sum(np.abs(y - yhat)),
        'centropy': lambda y, yhat: np.sum(yhat * np.log(y))
    }
    if name in j:
        return j[name]
    else:
        raise ValueError(f"'{name}' is an invalid loss function")


def d_losses(name):
    """ get derivative of raw cost function """
    dj = {
        'ssr': lambda y, yhat: -2 * (y - yhat),
        'sar': lambda y, yhat: -1 * np.sign(y - yhat),
        'centropy': lambda y, yhat: ...  # todo: write this loss algorithm
    }
    if name in dj:
        return dj[name]
    else:
        raise ValueError(f"'{name}' is an invalid loss function")


class Initializers:
    def __init__(self, method, params):
        self.methods = [
            'xavier',
            'gaussian',
            'zeros',
            'ones'
        ]
        self._method = self._check_method(method)
        self._params = self._get_params(params)
        self._initializer = self._get_initializer()

    def _check_method(self, method):
        if method in self.methods:
            return method
        else:
            # error
            raise ValueError(
                f"'{method}' is an invalid optimization technique\n"
                f"choose from: {[mtd for mtd in self.methods]}"
            )

    def _get_params(self, params):
        # define default params
        default = {
            'xavier': {
                'gain': 1
            },
            'gaussian': None,
            'zeros': None,
            'ones': None
        }

        if params == 'auto':
            # set hyperparams if auto
            self._params = default[self._method]
        elif isinstance(params, dict):
            # set default hyperparams
            self._params = default[self._method]
            # set hyperparams if defined
            for prm in params:
                if prm in self._params:
                    self._params[prm] = params[prm]
                else:
                    UserWarning(
                        f"{prm} is not a valid hyperparameter for {self._method}\n"
                        f"choose from: {[prm for prm in self._params]}"
                    )
        else:
            # error
            raise TypeError(
                f"hyperparameters must be a dictionary\n"
                f"choose from: {[prm for prm in default[self._method]]}"
            )

    def _get_initializer(self):
        # gaussian initialization
        def gaussian(row, col):
            np.random.randn(row, col)

        # xavier initialization
        def xavier(row, col):
            np.random.randn(row, col) * self._params['gain'] / np.sqrt(2 / (row + col))

        # zeros initialization
        def zeros(row, col):
            np.zeros((row, col))

        # ones initialization
        def ones(row, col):
            np.ones((row, col))

        init_funcs = {
            'gaussian': gaussian,
            'xavier': xavier,
            'zeros': zeros,
            'ones': ones
        }

        return init_funcs[self._method]

    def initialize(self, rows, columns):
        return self._initializer(rows, columns)


class Activators:
    ...


class Losses:
    ...


class Optimizers:
    """
    Optimization algorithms.
    """
    def __init__(self, algorithm: str, hyperparameters: dict = None):
        """
        'Optimizers' is a class containing various optimization algorithms.
        These optimization algorithms currently include 'adam' (Adam), 'sgd' (SGD), and 'rms' (RMSprop).
        The class structure promotes flexibility through the easy addition of other algorithms.

        Arguments:
            algorithm: A string referencing the optimization algorithm that will be pulled.
            hyperparameters: A dictionary referencing the hyperparameters for the optimization algorithm.
                (any hyperparameters not referenced will be automatically defined)
        """
        # optimization algorithms
        self.algorithms = [
            'adam',
            'sgd',
            'rms'
        ]

        # internal optimization algorithm parameters
        self._algorithm = self._check_algorithm(algorithm)
        self._hyps = self._get_hyperparams(hyperparameters)
        self._optim = self._get_optim()

        # internal memory
        self._memory = self._get_memory()

    def _check_algorithm(self, algorithm):
        # checks whether the optimization algorithm is valid
        if algorithm in self.algorithms:
            # valid optimization algorithm
            return algorithm
        else:
            # invalid optimization algorithm
            raise ValueError(
                f"Invalid optimization algorithm: '{algorithm}'\n"
                f"Choose from: {[alg for alg in self.algorithms]}"
            )

    def _get_hyperparams(self, hyperparams):
        # defines optimization algorithm hyperparameters
        # default optimization algorithm hyperparameters
        default = {
            'adam': {
                'gamma': 0.001,
                'lambda_d': 0,
                'beta': (0.9, 0.999),
                'epsilon': 1e-8,
                'ams': False
            },
            'sgd': {
                'gamma': 0.001,
                'lambda_d': 0,
                'mu': 0,
                'tau': 0,
                'nesterov': False
            },
            'rms': {
                'gamma': 0.001,
                'lambda_d': 0,
                'beta': 0.99,
                'mu': 0,
                'epsilon': 1e-8
            }
        }

        # instantiate hyperparameter dictionary
        hyps = default[self._algorithm]

        if hyperparams and isinstance(hyperparams, dict):
            # set defined hyperparameters
            for hyp in hyperparams:
                if hyp in hyps:
                    # todo: add errors for each hyperparameter
                    # valid hyperparameter
                    hyps[hyp] = hyperparams[hyp]
                else:
                    # invalid hyperparameter
                    raise UserWarning(
                        f"Invalid hyperparameter for {self._algorithm}: {hyp}\n"
                        f"Choose from: {[hyp for hyp in hyps]}"
                    )
        elif hyperparams:
            # invalid data type
            raise TypeError(
                f"'hyperparameters' is not a dictionary: {hyperparams}\n"
                f"Choose from: {[hyp for hyp in hyps]}"
            )

        # return hyperparameters
        return hyps

    def _get_optim(self):
        # defines optimization function
        # todo: check order of operations
        def adam(thetas, gradients):
            # Adam optimization algorithm
            # weight decay
            deltas = gradients + (self._hyps['lambda_d'] * thetas)
            if self._memory['deltas_p']:
                # momentum
                deltas = ((self._hyps['beta'][0] * self._memory['deltas_p']) + ((1 - self._hyps['beta'][0]) * deltas)) / (1 - self._hyps['beta'][0])
            if self._memory['upsilons_p']:
                # square momentum
                upsilons = ((self._hyps['beta'][1] * self._memory['upsilons_p']) / (1 - self._hyps['beta'][1])) + (deltas ** 2)
            else:
                # square
                upsilons = deltas ** 2
            if self._hyps['ams']:
                # ams-grad variant
                if self._memory['upsilons_mx']:
                    upsilons_mx = np.maximum(self._memory['upsilons_mx'], upsilons)
                else:
                    upsilons_mx = upsilons
                # update memory
                self._memory['upsilons_mx'] = upsilons_mx
                # set upsilons
                upsilons = upsilons_mx
            # optimization
            thetas -= self._hyps['gamma'] * deltas / (np.sqrt(upsilons) + self._hyps['epsilon'])

            # update memory
            self._memory['deltas_p'] = deltas
            self._memory['upsilons_p'] = upsilons

            # return thetas
            return thetas

        def sgd(thetas, gradients):
            # SGD optimization algorithm
            # weight decay
            deltas = gradients + (self._hyps['lambda_d'] * thetas)
            if self._hyps['mu'] and self._memory['deltas_p']:
                # momentum
                deltas = (self._hyps['mu'] * self._memory['deltas_p']) + ((1 - self._hyps['tau']) * deltas)
            if self._hyps['nesterov'] and self._hyps['mu']:
                # nesterov momentum
                deltas += gradients + self._hyps['mu'] * deltas
            # optimization
            thetas -= self._hyps['gamma'] * deltas

            # update memory
            self._memory['deltas_p'] = deltas

            # return thetas
            return thetas

        def rms(thetas, gradients):
            # RMSprop optimization algorithm
            # weight decay
            deltas = gradients + (self._hyps['lambda_d'] * thetas)
            if self._memory['upsilons_p']:
                # square momentum
                upsilons = (self._hyps['beta'] * self._memory['upsilons_p']) + ((1 - self._hyps['beta']) * (deltas ** 2))
            else:
                # square
                upsilons = (1 - self._hyps['beta']) * (deltas ** 2)
            # step calculation
            deltas = deltas / (np.sqrt(upsilons) + self._hyps['epsilon'])
            if self._hyps['mu'] and self._memory['deltas_p']:
                # momentum
                deltas += self._hyps['mu'] * self._memory['deltas_p']
            # optimization
            thetas -= self._hyps['gamma'] * deltas

            # update memory
            self._memory['deltas_p'] = deltas
            self._memory['upsilons_p'] = upsilons

            # return thetas
            return thetas

        # optimization algorithm dictionary
        optim_funcs = {
            'adam': adam,
            'sgd': sgd,
            'rms': rms
        }

        # return optimization algorithm
        return optim_funcs[self._algorithm]

    def _get_memory(self):
        # instantiates memory dictionary
        # required memory components for each optimization algorithm
        memories = {
            'adam': {
                'deltas_p': None,
                'upsilons_p': None,
                'upsilons_mx': None
            },
            'sgd': {
                'deltas_p': None
            },
            'rms': {
                'deltas_p': None,
                'upsilons_p': None
            }
        }
        # return memory dictionary
        return memories[self._algorithm]

    def update(self, thetas: np.ndarray, gradients: np.ndarray):
        """
        'update' is a built-in function in the 'Optimizers' class.
        This function updates the parameters of a model based on the gradients.

        Arguments:
            thetas: A numpy array of the parameters that will be optimized.
            gradients: A numpy array of the gradients used to optimize the parameters.

        Returns:
            An updated parameter.
        """
        if not isinstance(thetas, np.ndarray):
            # invalid datatype
            raise TypeError(f"'thetas' is not a numpy array: {thetas}")
        if not isinstance(gradients, np.ndarray):
            # invalid datatype
            raise TypeError(f"'gradients' is not a numpy array: {gradients}")

        # return updated thetas
        return self._optim(thetas, gradients)
