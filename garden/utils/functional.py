"""
mathematical functions
"""

import numpy as np


# todo: deprecate
def ssr(expected, predicted):
    """ calculate loss using the sum of the squared residuals """
    error = np.sum(np.subtract(expected, predicted) ** 2)
    return error


def xavier(length, width):
    """ initialize a random array using xavier initialization """
    return np.random.randn(length, width) * np.sqrt(2 / length)


def cost(name):
    if name == 'ssr':
        def function(expected, predicted):
            return np.sum(np.subtract(expected, predicted) ** 2)
        return function
    elif name == 'sar':
        def function(expected, predicted):
            return np.sum(np.abs(np.subtract(expected, predicted)))
        return function
    elif name == 'cross-entropy':
        # todo: the math is wrong on this and this doesn't work
        def function(expected, predicted):
            if not isinstance(expected, int):
                expected = np.nanargmax(expected)
            predicted = softmax(predicted)
            return -1 * np.log(predicted[expected])
        return function
    else:
        raise ValueError(f"'{name}' is an invalid loss function")


def d_cost(name):
    if name == 'ssr':
        def derivative(expected, predicted):
            return -2 * np.subtract(expected, predicted)
        return derivative
    elif name == 'sar':
        def derivative(expected, predicted):
            return -1 * np.sign(expected - predicted)
        return derivative
    elif name == 'cross-entropy':
        def derivative(expected, predicted):
            index = np.nanargmax(expected)
            derived = np.exp(predicted) / np.sum(np.exp(predicted))
            derived[index] = derived[index] - 1
            return derived
        return derivative
    else:
        raise ValueError(f"'{name}' is an invalid loss function")


# todo: move into activations and determine softmax'
def softmax(values):
    """ calculate outcome probabilities using softmax """
    return np.exp(values) / np.sum(np.exp(values))


def activations(name, beta=0.1):
    """ get raw activation functional """
    functions = {
        'relu': lambda x: np.maximum(0, x),
        'leaky relu': lambda x: np.maximum(beta * x, x),
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'tanh': lambda x: np.tanh(x),
        'softplus': lambda x: np.log(1 + np.exp(x)),
        'mish': lambda x: x * np.tanh(np.log(1 + np.exp(x)))
    }
    if name in functions:
        return functions[name]
    else:
        raise ValueError(f"'{name}' is an invalid activation function")


def d_activations(name, beta=0.1):
    """ get the derivatives of raw activation functional """
    derivatives = {
        'relu': lambda x: np.where(x > 0, 1, 0),
        'leaky relu': lambda x: np.where(x > 0, 1, beta),
        'sigmoid': lambda x: np.exp(-x) / ((1 + np.exp(-x)) ** 2),
        'tanh': lambda x: 1 - np.tanh(x) ** 2,
        'softplus': lambda x: 1 / (1 + np.exp(-x)),
        'mish': lambda x: np.tanh(np.log(1 + np.exp(x))) + (x * np.exp(x) - x * np.exp(x) * (np.tanh(np.log(1 + np.exp(x)))) ** 2) / (1 + np.exp(x))
    }
    if name in derivatives:
        return derivatives[name]
    else:
        raise ValueError(f"'{name}' is an invalid activation function")
