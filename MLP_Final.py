import ast
import math
import random
import time

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm


class MLP:
    def __init__(self, load=False, weights="none", biases="none", layer_sizes=[784, 100, 10], activation="ReLU"):
        self.version = "1.6"
        self.load = load
        self.weights, self.biases = self._get_parameters(weights, biases)
        self.layer_sizes = layer_sizes
        self.activation = self._get_activation(activation)
        self.graph_results = False

    def _get_parameters(self, weights, biases):
        if weights == "none":
            weights = []
            for i in range(len(self.layer_sizes) - 1):
                weights.append(self.xavier_initialize(self.layer_sizes[i], self.layer_sizes[i + 1]))
        else:
            weights = weights
        if biases == "none":
            biases = []
            for i in range(len(self.layer_sizes) - 1):
                biases.append(np.zeros((1, self.layer_sizes[i + 1])))
        else:
            biases = biases
        return weights, biases

    def _get_activation(self, name):
        if name == "Sigmoid":
            return {
                "forward": lambda x: 1 / (1 + np.exp(-x)),
                "derivative": lambda x: x * (1 - x)
            }
        elif name == "Tanh":
            return {
                "forward": lambda x: np.tanh(x),
                "derivative": lambda x: 1 - np.tanh(x) ** 2
            }
        elif name == "ReLU":
            return {
                "forward": lambda x: np.maximum(0, x),
                "derivative": lambda x: np.where(x > 0, 1, 0)
            }
        elif name == "Leaky ReLU":
            return {
                "forward": lambda x: np.maximum(0.1 * x, x),
                "derivative": lambda x: np.where(x > 0, 1, 0.1)
            }
        else:
            raise ValueError(f'"{name}" is an invalid activation function.')

    def _get_solver(self, name):
        if name == "SGD":
            def update(nodes, Y, weights, biases):
                d_weights = []
                d_biases = []

                d_b = -2 * (Y - nodes[-1])
                d_biases.insert(0, d_b)
                for layer in range(-1, -len(nodes) + 1, -1):
                    d_w = nodes[layer - 1].T * d_b
                    d_weights.insert(0, d_w)
                    d_b = np.array([np.sum(weights[layer] * d_b, axis=1)])
                    d_biases.insert(0, d_b)
                d_w = nodes[0].T * d_b
                d_weights.insert(0, d_w)

                for layer in range(len(nodes) - 1):
                    weights[layer] -= learning_rate * (d_weights[layer] + (alpha / train_len) * weights[layer])
                    biases[layer] -= learning_rate * d_biases[layer]

                return weights, biases
        elif name == "Mini-Batch":
            def update():
                ...
        else:
            raise ValueError(f"'{name}' is an invalid solving method.")

    def xavier_initialize(self, length, width):
        array = np.random.randn(length, width) * math.sqrt(2 / length)
        return array

    def forward(self, inputs):
        nodes = [inputs]
        for layer in range(len(self.layer_sizes) - 1):
            node_layer = self.activation["forward"](np.matmul(nodes[-1], self.weights[layer]) + self.biases[layer])  # consider renaming
            nodes.append(node_layer)
        return nodes

    def backward(self, values, predicted, expected):
        ...

    def predict(self, inputs):
        predicted = np.nanargmax(self.forward(inputs)[-1])
        return predicted

    def set_output_configuration(self, graph_results=False, cm_normalization="true", eval_batching=True, eval_batch_size=5, eval_interval=10):
        self.graph_results = graph_results
        self.cm_normalization = cm_normalization
        self.eval_batching = eval_batching
        self.eval_batch_size = eval_batch_size
        self.eval_interval = eval_interval

    def fit(self, X, Y, solver="Mini-Batch", alpha=0.1, batch_size="auto", learning_rate=0.001, max_iter=200, momentum=0.9):
        self.solver = self._get_solver(solver)
        train_len = len(X)
        for epoch in range(max_iter):
            tc = random.randint(0, train_len - 1)
            inputs = X[tc]
            expected = Y[tc]
            nodes = self.forward()
            self.weights, self.biases = self.backward()


# loading data for testing
from PIL import Image
keras_weights = []
keras_biases = []
with open(f"saved/weights_keras.txt", "r") as f:
    for line in f:
        keras_weights.append(np.array(ast.literal_eval(line)))
with open(f"saved/biases_keras.txt", "r") as f:
    for line in f:
        keras_biases.append(np.array(ast.literal_eval(line)))
img = Image.open(f"saved/user_number.jpeg")
gray_img = img.convert("L")
test_input = np.array(list(gray_img.getdata())) / 255

# class testing
neural_net = MLP(weights=keras_weights, biases=keras_biases, layer_sizes=[784, 16, 16, 10], activation="Leaky ReLU")
print(neural_net.forward(test_input)[-1])
print(neural_net.predict(test_input))
