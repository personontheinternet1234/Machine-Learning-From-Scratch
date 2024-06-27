"""
mathematical functions
"""

import numpy as np


def ssr(expected, predicted):
    """ calculate loss using the sum of the squared residuals """
    error = np.sum(np.subtract(expected, predicted) ** 2)
    return error


def loss(name):
    if name == 'ssr':
        def forward(expected, predicted):
            return np.sum(np.subtract(expected, predicted) ** 2)

        def backward():
            ...


def softmax(values):
    """ calculate outcome probabilities using softmax """
    output = np.exp(values) / np.sum(np.exp(values))
    return output


def xavier_initialize(length, width):
    """ initialize a random array using xavier initialization """
    array = np.random.randn(length, width) * np.sqrt(2 / length)
    return array


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
        possible_functions = []
        for key, value in functions.items():
            possible_functions.append(key)
        raise ValueError(
            f"'{name}' is an invalid activation function, "
            f"choose from: {possible_functions}"
        )


def derivative_activations(name, beta=0.1):
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
        possible_functions = []
        for key, value in derivatives.items():
            possible_functions.append(key)
        raise ValueError(
            f"'{name}' is an invalid activation function, "
            f"choose from: {possible_functions}"
        )
