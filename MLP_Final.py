import ast
import math
import random
import time

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm


def l_relu(values):
    return {
        "function": np.maximum(0.1 * values, values),
        "derivative": np.where(values > 0, 1, 0.1)
    }

def relu(values):
    output = np.maximum(0, values)
    return output


def sigmoid(values):
    output = 1 / (1 + np.exp(-1 * values))
    return output


def tanh(values):
    output = np.tanh(values)
    return output


activations = {
    "Leaky ReLU": l_relu,
    "ReLU": relu,
    "Sigmoid": sigmoid,
    "Tanh": tanh
}


class MLP:
    def __init__(self, weights, biases, hidden_layer_sizes):
        self.weights = weights
        self.biases = biases
        self.hidden_layer_sizes = hidden_layer_sizes

    def forward(self, inputs, activation):
        self.activation = activations[activation]

    def backward(self, solver):
        if solver == "Mini-batch":
            ...
        elif solver == "Stochastic":
            ...