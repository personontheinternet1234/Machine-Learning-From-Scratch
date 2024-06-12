"""
Applied functions
Authors:
    Christian SW Host-Madsen CO '25 <chost-madsen25@punahou.edu>
    Isaac Park Verbrugge CO '25 <iverbrugge25@punahou.edu>
"""

import math
import random

import numpy as np


def ssr(expected, predicted):
    """ calculate loss using the sum of the squared residuals """
    loss = np.sum(np.subtract(expected, predicted) ** 2)
    return loss


def softmax(values):
    """ calculate outcome probabilities using softmax """
    output = np.exp(values) / np.sum(np.exp(values))
    return output


def xavier_initialize(length, width):
    """ initialize a random array using xavier initialization """
    array = np.random.randn(length, width) * math.sqrt(2 / length)
    return array


def activations(name):
    """ get raw activation functions """
    functions = {
        'relu': lambda x: np.maximum(0, x),
        'leaky relu': lambda x: np.maximum(0.1 * x, x),
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'tanh': lambda x: np.tanh(x),
        'softplus': lambda x: math.log(1 + np.exp(x))
    }
    if name in functions:
        return functions[name]
    else:
        raise ValueError(f"'{name}' is an invalid activation function")


def derivative_activations(name):
    """ get the derivatives of raw activation functions """
    derivatives = {
        'relu': lambda x: np.where(x > 0, 1, 0),
        'leaky relu': lambda x: np.where(x > 0, 1, 0.1),
        'sigmoid': lambda x: np.exp(-x) / ((1 + np.exp(-x)) ** 2),
        'tanh': lambda x: 1 - np.tanh(x) ** 2,
        'softplus': lambda x: 1 / (1 + np.exp(-x))
    }
    if name in derivatives:
        return derivatives[name]
    else:
        raise ValueError(f"'{name}' is an invalid activation function")


def trim(data, trim_frac=0.5):
    """ trim data based on a fraction """
    data = data[0:trim_frac * len(data)]
    return data


def test_val(values, labels, val_frac=0.3):
    """ randomly split data into training and validation sets """
    # zip & shuffle data
    data = list(zip(values, labels))
    random.shuffle(data)
    # split data
    train = data[round(len(data) * val_frac):]
    val = data[0:round(len(data) * val_frac)]
    # unzip data
    train_values, train_labels = zip(*train)
    val_values, val_labels = zip(*val)
    # reformat data
    train_values, train_labels = list(train_values), list(train_labels)
    val_values, val_labels = list(val_values), list(val_labels)
    # return data
    return train_values, train_labels, val_values, val_labels
