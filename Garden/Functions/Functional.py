"""
Applied functions
Authors:
    Christian SW Host-Madsen CO '25 <chost-madsen25@punahou.edu>
    Isaac Park Verbrugge CO '25 <iverbrugge25@punahou.edu>
"""


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
