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
    """ initialize a random array using xavier initialization """
    return np.random.randn(length, width) * np.sqrt(2 / length)


# todo: move into activations
def softmax(values):
    """ calculate outcome probabilities using softmax """
    return np.exp(values) / np.sum(np.exp(values))


def activations(name, beta=0.1):
    """ get raw activation function """
    g = {
        'softmax': lambda x: np.exp(x) / np.sum(np.exp(x)),
        'relu': lambda x: np.maximum(0, x),
        'leaky relu': lambda x: np.maximum(beta * x, x),
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'tanh': lambda x: np.tanh(x),
        'softplus': lambda x: np.log(1 + np.exp(x)),
        'mish': lambda x: x * np.tanh(np.log(1 + np.exp(x)))
    }
    if name in g:
        return g[name]
    else:
        raise ValueError(f"'{name}' is an invalid activation function")


def d_activations(name, beta=0.1):
    """ get derivative of raw activation function """
    dg = {
        'softmax': lambda x: x,  # todo
        'relu': lambda x: np.where(x > 0, 1, 0),
        'leaky relu': lambda x: np.where(x > 0, 1, beta),
        'sigmoid': lambda x: np.exp(-x) / ((1 + np.exp(-x)) ** 2),
        'tanh': lambda x: 1 - np.tanh(x) ** 2,
        'softplus': lambda x: 1 / (1 + np.exp(-x)),
        'mish': lambda x: np.tanh(np.log(1 + np.exp(x))) + (x * np.exp(x) - x * np.exp(x) * (np.tanh(np.log(1 + np.exp(x)))) ** 2) / (1 + np.exp(x))
    }
    if name in dg:
        return dg[name]
    else:
        raise ValueError(f"'{name}' is an invalid activation function")


def cost(name):
    """ get raw cost function """
    if name == 'ssr':
        def j(expected, predicted):
            return np.sum(np.subtract(expected, predicted) ** 2)
        return j
    elif name == 'sar':
        def j(expected, predicted):
            return np.sum(np.abs(np.subtract(expected, predicted)))
        return j
    elif name == 'cross-entropy':
        # todo
        def j(expected, predicted):
            ...
        return j
    else:
        raise ValueError(f"'{name}' is an invalid loss function")


def d_cost(name):
    """ get derivative of raw cost function """
    if name == 'ssr':
        def dj(expected, predicted):
            return -2 * np.subtract(expected, predicted)
        return dj
    elif name == 'sar':
        def dj(expected, predicted):
            return -1 * np.sign(expected - predicted)
        return dj
    elif name == 'cross-entropy':
        def dj(expected, predicted):
            # todo
            ...
        return dj
    else:
        raise ValueError(f"'{name}' is an invalid loss function")


def optimizer(name):
    if name == 'adam':
        def optim(gradient, gamma=0.001, betas=(0.9, 0.999), lambda_d=0, epsilon=1e-8):
            ...
        return optim
    elif name == 'gradient descent':
        def optim(thetas, gradients, gamma=0.001, mu=0, tau=0, lambda_d=0):
            for t in range(len(thetas)):
                delta = gradients[t]
                if lambda_d:
                    delta = delta + (lambda_d * thetas[t])
                if mu and t:
                    delta = (mu * gradients[t-1]) + ((1 - tau) * delta)
                thetas[t] -= gamma * delta
            return thetas
        return optim
    elif name == 'rmsprop':
        def optim(thetas, gradients, gamma=0.01, alpha=0.99, mu=0, lambda_d=0, epsilon=1e-8):
            v = 0
            b = 0
            for t in range(len(thetas)):
                delta = gradients[t]
                if lambda_d:
                    delta = delta + (lambda_d * thetas[t])
                v = (alpha * v) + (1 - alpha) * (delta ** 2)
                if mu:
                    delta = (mu * b) + (delta / (math.sqrt(v) + epsilon))
                    b = delta
                    thetas[t] -= gamma * delta
                else:
                    thetas[t] -= gamma * (delta / (math.sqrt(v) + epsilon))
        return optim
    else:
        raise ValueError(f"'{name}' is an invalid optimizer")

