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
    def __init__(self, weights, biases, layer_sizes):
        self.weights = weights
        self.biases = biases
        self.layer_sizes = layer_sizes
        self.nodes = []

    def _get_activation(self):
        ...

    def forward(self, inputs):
        working_nodes = [inputs]
        for layer in range(len(self.layer_sizes) - 1):
            activations = l_relu(np.matmul(working_nodes[-1], self.weights[layer]) + self.biases[layer])  # consider renaming
            working_nodes.append(activations)
        self.nodes = working_nodes

    def sgd_backward(self, expected, learning_rate, alpha, train_len):
        # initialize gradient lists
        d_weights = []
        d_biases = []

        d_b = -2 * (expected - self.nodes[-1])
        d_biases.insert(0, d_b)
        for layer in range(-1, -len(self.nodes) + 1, -1):
            d_w = self.nodes[layer - 1].T * d_b
            d_weights.insert(0, d_w)
            d_b = np.array([np.sum(self.weights[layer] * d_b, axis=1)])
            d_biases.insert(0, d_b)
        d_w = self.nodes[0].T * d_b
        d_weights.insert(0, d_w)

        # haven't tried it yet-- this code could cause problems.
        for layer in range(len(self.nodes) - 1):
            self.weights[layer] -= learning_rate * (d_weights[layer] + (alpha / train_len) * self.weights[layer])
            self.biases[layer] -= learning_rate * d_biases[layer]

    def fit(self, data_samples_and_answers, max_iter, solver, learning_rate, alpha, momentum, train_len, batch_size=200):
        train_len = len(data_samples_and_answers)

        if solver == "Mini-batch":
            ...
        elif solver == "Stochastic":
            for i in range(len(max_iter)):
                # replace data_samples_and_answers[0], [1] with whatever the real answers are yk
                self.forward(self, data_samples_and_answers[0])
                self.sgd_backward(data_samples_and_answers[1], learning_rate, alpha, train_len)

neural_net = MLP()

neural_net.forward()
neural_net.backward()