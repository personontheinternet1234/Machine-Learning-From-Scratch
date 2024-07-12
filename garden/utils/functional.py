"""
mathematical functions
"""

import numpy as np
import math


# todo: deprecate
def ssr(expected, predicted):
    c = np.sum(np.subtract(expected, predicted) ** 2)
    return c


def xavier(length, width):
    # todo: deprecate
    """ initialize a random array using xavier initialization """
    return np.random.randn(length, width) * np.sqrt(2 / length)


def softmax(x):
    # todo: deprecate
    """ calculate outcome probabilities using softmax """
    return np.exp(x) / np.sum(np.exp(x))


def initialize(name):
    init = {
        'gaussian': lambda l, w: np.random.randn(l, w),
        'xavier': lambda l, w: np.random.randn(l, w) * np.sqrt(2 / w)  # todo: check if correct
        # zeros and ones later
    }
    if name in init:
        return init[name]
    else:
        raise ValueError(f"'{name}' is an invalid initialization method")


def activations(name, beta='auto'):
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
            beta = 0
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


def d_activations(name, beta=0.1):
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
        'softmax': lambda x: x,  # todo
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


def cost(name):
    """ get raw cost function """
    j = {
        'mse': lambda y, yhat: np.sum((y - yhat) ** 2),
        'l1': lambda y, yhat: np.sum(np.abs(y - yhat)),
        'cross-entropy': lambda y, yhat: ...
    }
    if name in j:
        return j[name]
    else:
        raise ValueError(f"'{name}' is an invalid cost function")


def d_cost(name):
    """ get derivative of raw cost function """
    # todo: restructure to lambda functions
    if name == 'mse':
        def dj(expected, predicted):
            return -2 * np.subtract(expected, predicted)
        return dj
    elif name == 'l1':
        def dj(expected, predicted):
            return -1 * np.sign(expected - predicted)
        return dj
    elif name == 'cross-entropy':
        def dj(expected, predicted):
            # todo
            ...
        return dj
    else:
        raise ValueError(f"'{name}' is an invalid cost function")


def optimizer(name):
    # todo: check how well python respects the order of operations
    if name == 'adam':
        def optim(gradient, gamma=0.001, betas=(0.9, 0.999), lambda_d=0, epsilon=1e-8):
            ...
        return optim
    elif name == 'gradient descent':
        def optim(thetas, gradients, deltas_p=None, gamma=0.001, mu=0, tau=0, lambda_d=0):
            deltas = gradients + (lambda_d * thetas)
            if mu and deltas_p:
                deltas = (mu * deltas_p) + ((1 - tau) * deltas)
            thetas -= gamma * deltas
            return thetas, deltas
        return optim
    elif name == 'rmsprop':
        def optim(thetas, gradients, upsilons_p=None, deltas_p=None, gamma=0.01, alpha=0.99, mu=0, lambda_d=0, epsilon=1e-8):
            deltas = gradients + (lambda_d * thetas)
            if upsilons_p:
                upsilons = (alpha * upsilons_p) + ((1 - alpha) * (deltas ** 2))
            else:
                upsilons = (1 - alpha) * (deltas ** 2)

            if mu and deltas_p:
                deltas = (mu * deltas_p) + ...
            else:
                thetas -= gamma * deltas / (math.sqrt(upsilons) + epsilon)
            return thetas, deltas, upsilons

            # v = 0
            # b = 0
            # for t in range(len(thetas)):
            #     delta = gradients[t]
            #     if lambda_d:
            #         delta = delta + (lambda_d * thetas[t])
            #     v = (alpha * v) + (1 - alpha) * (np.sum(delta) ** 2)
            #     if mu:
            #         delta = (mu * b) + (delta / (math.sqrt(v) + epsilon))
            #         b = delta
            #         thetas[t] -= gamma * delta
            #     else:
            #         thetas[t] -= gamma * delta / (math.sqrt(v) + epsilon)
        return optim
    else:
        raise ValueError(f"'{name}' is an invalid optimizer")
