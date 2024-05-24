import ast
import math
import random
import time

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from PIL import Image

keras_weights = []
keras_biases = []
with open(f"saved/weights_keras.txt", "r") as f:
    for line in f:
        keras_weights.append(np.array(ast.literal_eval(line)))
with open(f"saved/biases_keras.txt", "r") as f:
    for line in f:
        keras_biases.append(np.array(ast.literal_eval(line)))


class MLP:
    def __init__(self, load=False, weights=[], biases=[], layer_sizes=[100], activation="ReLU"):
        self.version = "1.5"
        self.load = load
        self.weights = weights
        self.biases = biases
        self.layer_sizes = layer_sizes
        self.activation = self._get_activation(activation)
        self.solver = self._get_solver(solver)
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.momentum = momentum

    def _get_activation(self, name):
        if name == "Sigmoid":
            return {
                "normal": lambda x: 1 / (1 + np.exp(-x)),
                "derivative": lambda x: x * (1 - x)
            }
        elif name == "Tanh":
            return {
                "normal": lambda x: np.tanh(x),
                "derivative": lambda x: (np.cosh(x)) ** -2
            }
        elif name == "ReLU":
            return {
                "normal": lambda x: np.maximum(0, x),
                "derivative": lambda x: np.where(x > 0, 1, 0)
            }
        elif name == "Leaky ReLU":
            return {
                "normal": lambda x: np.maximum(0.1 * x, x),
                "derivative": lambda x: np.where(x > 0, 1, 0.1)
            }
        else:
            raise ValueError(f"[{name}] is an invalid activation function.")

    def _get_solver(self, name):

    def forward(self, inputs):
        nodes = [inputs]
        for layer in range(len(self.layer_sizes) - 1):
            node_layer = self.activation_function(activation_name)["normal"](np.matmul(nodes[-1], self.weights[layer]) + self.biases[layer])  # consider renaming
            nodes.append(node_layer)
        return nodes

    def predict(self, inputs):
        predicted = np.nanargmax(self.forward(inputs)[-1])
        return predicted

    def sgd_backward(self, values, predicted, expected):
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

    def fit(self, values, labels, solver="Mini-Batch", alpha=0.1, batch_size="auto", learning_rate=0.001, max_iter=200, momentum=0.9):
        for epoch in range(self.max_iter):
            nodes = self.forward()


neural_net = MLP(weights=keras_weights, biases=keras_biases, layer_sizes=[784, 16, 16, 10])

img = Image.open(f"saved/user_number.jpeg")
# grayscale image
gray_img = img.convert("L")
# convert to numpy array
forward_layer = np.array(list(gray_img.getdata())) / 255

print(neural_net.forward(forward_layer, activation_name="LeakyReLU")[-1])

print(neural_net.predict(forward_layer))
