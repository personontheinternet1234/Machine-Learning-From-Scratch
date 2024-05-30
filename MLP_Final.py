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

    def __init__(self, load=False, weights='none', biases='none', layer_sizes='auto', activation='relu'):
        self.version = '1.6'
        self.load = load
        self.weights, self.biases = self.set_parameters(weights, biases)
        if layer_sizes == 'auto':
            self.layer_sizes = [784, 100, 10]
        else:
            self.layer_sizes = layer_sizes
        self.activation = self._get_activation(activation)
        self.X = None
        self.Y = None
        self.valid_X = None
        self.valid_Y = None
        self.solver = False
        self.batch_size = None
        self.train_len = 0
        self.valid_len = 0
        self.loss_reporting = False
        self.eval_batching = False
        self.eval_batch_size = None
        self.eval_interval = None
        self.train_losses = None
        self.valid_losses = None
        self.start_time = None
        self.end_time = None

    def set_parameters(self, weights, biases):
        if weights == 'none':
            weights = []
            for i in range(len(self.layer_sizes) - 1):
                weights.append(self.xavier_initialize(self.layer_sizes[i], self.layer_sizes[i + 1]))
        else:
            weights = weights
        if biases == 'none':
            biases = []
            for i in range(len(self.layer_sizes) - 1):
                biases.append(np.zeros((1, self.layer_sizes[i + 1])))
        else:
            biases = biases
        return weights, biases

    def xavier_initialize(self, length, width):
        array = np.random.randn(length, width) * math.sqrt(2 / length)
        return array

    def _get_activation(self, name):
        if name == 'sigmoid':
            return {
                'forward': lambda x: 1 / (1 + np.exp(-x)),
                'derivative': lambda x: x * (1 - x)
            }
        elif name == 'tanh':
            return {
                'forward': lambda x: np.tanh(x),
                'derivative': lambda x: 1 - np.tanh(x) ** 2
            }
        elif name == 'relu':
            return {
                'forward': lambda x: np.maximum(0, x),
                'derivative': lambda x: np.where(x > 0, 1, 0)
            }
        elif name == 'leaky relu':
            return {
                'forward': lambda x: np.maximum(0.1 * x, x),
                'derivative': lambda x: np.where(x > 0, 1, 0.1)
            }
        else:
            raise ValueError(f"'{name}' is an invalid activation function")

    def _get_solver(self, name):
        if name == 'sgd':
            def update(nodes, Y, weights, biases, learning_rate, alpha, momentum):
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
                    weights[layer] -= learning_rate * (d_weights[layer] + (alpha / self.train_len) * weights[layer])
                    biases[layer] -= learning_rate * d_biases[layer]

                return weights, biases

            return {'update': update}
        elif name == 'mini-batch':
            def update(nodes, Y, weights, biases, learning_rate, alpha, momentum):
                d_weights = []
                d_biases = []

                d_b = -2 * (Y - nodes[-1])
                d_biases.insert(0, d_b)
                for layer in range(-1, -len(nodes) + 1, -1):
                    d_w = np.reshape(nodes[layer - 1], (self.batch_size, self.layer_sizes[layer - 1], 1)) * d_b
                    d_weights.insert(0, d_w)
                    d_b = np.reshape(np.array([np.sum(weights[layer] * d_b, axis=2)]), (self.batch_size, 1, self.layer_sizes[layer - 1]))
                    d_biases.insert(0, d_b)
                d_w = np.reshape(nodes[0], (self.batch_size, self.layer_sizes[0], 1)) * d_b
                d_weights.insert(0, d_w)

                for layer in range(len(nodes) - 1):
                    weights[layer] -= learning_rate * np.sum((d_weights[layer] + (alpha / self.batch_size) * weights[layer]), axis=0) / self.batch_size
                    biases[layer] -= learning_rate * np.sum(d_biases[layer], axis=0) / self.batch_size

                return weights, biases

            return {'update': update}
        else:
            raise ValueError(f"'{name}' is an invalid solving method")

    def forward(self, inputs):
        nodes = [inputs]
        for layer in range(len(self.layer_sizes) - 1):
            node_layer = self.activation['forward'](np.matmul(nodes[-1], self.weights[layer]) + self.biases[layer])
            nodes.append(node_layer)
        return nodes

    def loss(self, expected, predicted):
        loss = np.sum(np.subtract(expected, predicted) ** 2)
        return loss

    def predict(self, inputs):
        predicted = np.nanargmax(self.forward(inputs)[-1])
        return predicted

    def set_valid(self, valid_X, valid_Y):
        self.valid_X = valid_X
        self.valid_Y = valid_Y
        self.valid_len = len(self.valid_X)

    def fit(self, X, Y, solver='mini-batch', alpha=0.1, batch_size='auto', learning_rate=0.001, max_iter=2000, momentum=0.9):
        self.solver = self._get_solver(solver)
        self.X = X
        self.Y = Y
        self.train_len = len(self.X)
        if batch_size == 'auto':
            self.batch_size = min(20, len(self.X))
        elif (not isinstance(batch_size, int)) or batch_size <= 1:
            raise ValueError(f"'{batch_size}' isn't a valid batch size")
        else:
            self.batch_size = batch_size
        if self.eval_batch_size == 'auto':
            self.eval_batch_size = min(20, len(self.X))
        train_losses = []
        valid_losses = []
        self.start_time = time.time()
        for epoch in tqdm(range(max_iter), ncols=100, desc='fitting'):
            if solver == 'mini-batch':
                tc = random.randint(self.batch_size, self.train_len)
                inputs = array_X[tc - self.batch_size:tc]
                expected = array_Y[tc - self.batch_size:tc]
            elif solver == 'sgd':
                tc = random.randint(0, self.train_len - 1)
                inputs = X[tc]
                expected = Y[tc]
            nodes = self.forward(inputs)
            self.weights, self.biases = self.solver['update'](nodes, expected, self.weights, self.biases, learning_rate, alpha, momentum)
            if self.loss_reporting and epoch % self.eval_interval == 0:
                tc = random.randint(self.eval_batch_size, self.train_len)
                valid_tc = random.randint(self.eval_batch_size, self.valid_len)
                train_pred = self.forward(self.X[tc - self.eval_batch_size:tc])[-1]
                valid_pred = self.forward(self.valid_X[valid_tc - self.eval_batch_size:valid_tc])[-1]
                train_loss = self.loss(train_pred, self.Y[tc - self.eval_batch_size:tc])
                valid_loss = self.loss(valid_pred, self.valid_Y[valid_tc - self.eval_batch_size:valid_tc])
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
        self.end_time = time.time()

        self.train_losses = train_losses
        self.valid_losses = valid_losses

    def results(self):
        loss = np.sum(np.subtract(self.Y, self.forward(X)[-1]) ** 2) / self.train_len
        val_loss = np.sum(np.subtract(self.valid_Y, self.forward(self.valid_X)[-1]) ** 2) / self.valid_len
        accu = 0
        for i in tqdm(range(self.train_len), ncols=100, desc='calculating train accu'):
            predicted = self.forward(self.X[i])[-1]
            expected = self.Y[i]
            if np.nanargmax(predicted) == np.nanargmax(expected):
                accu += 1
        accu /= self.train_len
        val_accu = 0
        for i in tqdm(range(self.valid_len), ncols=100, desc='calculating val accu'):
            predicted = self.forward(self.valid_X[i],)[-1]
            expected = self.valid_Y[i]
            if np.nanargmax(predicted) == np.nanargmax(expected):
                val_accu += 1
        val_accu /= self.valid_len
        elapsed_time = self.end_time - self.start_time

        return loss, val_loss, accu, val_accu, elapsed_time


    def configure_reporting(self, loss_reporting=False, eval_batching=True, eval_batch_size="auto", eval_interval=10):
        self.loss_reporting = loss_reporting
        self.eval_batching = eval_batching
        if eval_batch_size == 'auto':
            self.eval_batch_size = 'auto'
        elif (not isinstance(eval_batch_size, int)) or eval_batch_size <= 0:
            raise ValueError(f"'{eval_batch_size}' isn't a valid evaluation batch size")
        else:
            self.eval_batch_size = eval_batch_size
        self.eval_interval = eval_interval


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

# load MNIST data
df_values = np.array(pd.read_csv(f"data/data_values_keras.csv")).tolist()
for i in tqdm(range(len(df_values)), ncols=150, desc="Reformatting Data Values"):
    df_values[i] = np.array([df_values[i]])
df_labels = np.array(pd.read_csv(f"data/data_labels_keras.csv")).tolist()
for i in tqdm(range(len(df_labels)), ncols=150, desc="Reformatting Data Labels"):
    df_labels[i] = np.array([df_labels[i]])


def test_train_split(data, test_size):
    random.shuffle(data)
    test = data[0:round(len(data) * test_size)]
    train = data[round(len(data) * test_size):]
    return train, test


# split training and testing data
train, test = test_train_split(list(zip(df_values, df_labels)), test_size=0.3)
# unzip training and testing data
X, Y = zip(*train)
X_test, Y_test = zip(*test)
X, Y = list(X), list(Y)
# reformat training and testing data
X_test, Y_test = list(X_test), list(Y_test)
array_X, array_Y = np.array(X), np.array(Y)
array_X_test, array_Y_test = np.array(X_test), np.array(Y_test)

# class testing
neural_net = MLP(weights=keras_weights, biases=keras_biases, layer_sizes=[784, 16, 16, 10], activation="leaky relu")
neural_net.fit(X, Y, solver="mini-batch")
print(neural_net.results)
print(neural_net.forward(test_input)[-1])
print(neural_net.predict(test_input))
