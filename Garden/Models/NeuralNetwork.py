import math
import random
import time

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from tqdm import tqdm


class NeuralNetwork:

    def __init__(self, weights='none', biases='none', layer_sizes='auto', activation='relu'):
        self.version = '1.7'
        if layer_sizes == 'auto':
            self.layer_sizes = [784, 100, 10]
        elif not isinstance(layer_sizes, list):
            raise ValueError(f"'{layer_sizes}' is not a list")
        elif len(layer_sizes) <= 2:
            raise ValueError(f"'{layer_sizes}' does not have the valid amount of layer sizes (2)")
        elif not all(isinstance(i, int) for i in layer_sizes):
            raise ValueError(f"'{layer_sizes}' must only include integers")
        else:
            self.layer_sizes = layer_sizes
        self.weights, self.biases = self.set_parameters(weights, biases)
        self.activation = self._get_activation(activation)
        self.X = None
        self.Y = None
        self.array_X = None
        self.array_Y = None
        self.set_valid = False
        self.valid_X = None
        self.valid_Y = None
        self.array_valid_X = None
        self.array_valid_Y = None
        self.solver = False
        self.batch_size = None
        self.alpha = None
        self.learning_rate = None
        self.max_iter = None
        self.train_len = None
        self.valid_len = None
        self.loss_reporting = False
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
                # weights.append(self.xavier_initialize(self.layer_sizes[i], self.layer_sizes[i + 1]))
                weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * math.sqrt(2 / self.layer_sizes[i]))
        else:
            weights = weights
        if biases == 'none':
            biases = []
            for i in range(len(self.layer_sizes) - 1):
                biases.append(np.zeros((1, self.layer_sizes[i + 1])))
        else:
            biases = biases
        return weights, biases

    # def xavier_initialize(self, length, width):
    #     array = np.random.randn(length, width) * math.sqrt(2 / length)
    #     return array

    @staticmethod
    def softmax(values):
        output = np.exp(values) / np.sum(np.exp(values))
        return output

    @staticmethod
    def _get_activation(name):
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
                    d_b = np.reshape(np.array([np.sum(weights[layer] * d_b, axis=2)]),
                                     (self.batch_size, 1, self.layer_sizes[layer - 1]))
                    d_biases.insert(0, d_b)
                d_w = np.reshape(nodes[0], (self.batch_size, self.layer_sizes[0], 1)) * d_b
                d_weights.insert(0, d_w)

                for layer in range(len(nodes) - 1):
                    weights[layer] -= learning_rate * np.sum(
                        (d_weights[layer] + (alpha / self.batch_size) * weights[layer]), axis=0) / self.batch_size
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

    @staticmethod
    def loss(expected, predicted):
        loss = np.sum(np.subtract(expected, predicted) ** 2)
        return loss

    def predict(self, inputs):
        predicted = np.nanargmax(self.forward(inputs)[-1])
        return predicted

    def valid(self, valid_X, valid_Y):
        self.set_valid = True
        self.valid_X = valid_X
        self.valid_Y = valid_Y
        self.array_valid_X = np.array(valid_X)
        self.array_valid_Y = np.array(valid_Y)
        self.valid_len = len(self.valid_X)

    def fit(self, X, Y, solver='mini-batch', alpha=0.0001, batch_size='auto', learning_rate=0.001, max_iter=20000,
            momentum=0.9):
        self.solver = self._get_solver(solver)
        self.X = X
        self.Y = Y
        self.array_X = np.array(X)
        self.array_Y = np.array(Y)
        self.train_len = len(self.X)
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        if batch_size == 'auto':
            self.batch_size = min(20, len(self.X))
        elif (not isinstance(batch_size, int)) or batch_size <= 1:
            raise ValueError(f"'{batch_size}' is an invalid batch size")
        else:
            self.batch_size = batch_size
        if self.eval_batch_size == 'auto' and self.set_valid:
            self.eval_batch_size = min(20, len(self.valid_X))
        elif self.eval_batch_size == 'auto':
            self.eval_batch_size = min(20, len(self.X))
        train_losses = []
        valid_losses = []
        self.start_time = time.time()
        for iter in tqdm(range(max_iter), ncols=100, desc='fitting'):
            if solver == 'mini-batch':
                tc = random.randint(self.batch_size, self.train_len)
                inputs = self.array_X[tc - self.batch_size:tc]
                expected = self.array_Y[tc - self.batch_size:tc]
            elif solver == 'sgd':
                tc = random.randint(0, self.train_len - 1)
                inputs = X[tc]
                expected = Y[tc]
            nodes = self.forward(inputs)
            self.weights, self.biases = self.solver['update'](nodes, expected, self.weights, self.biases, learning_rate, alpha, momentum)
            if self.loss_reporting and iter % self.eval_interval == 0:
                tc = random.randint(self.eval_batch_size, self.train_len)
                train_pred = self.forward(self.X[tc - self.eval_batch_size:tc])[-1]
                train_loss = self.loss(train_pred, self.Y[tc - self.eval_batch_size:tc])
                train_losses.append(train_loss)
                if self.set_valid:
                    valid_tc = random.randint(self.eval_batch_size, self.valid_len)
                    valid_pred = self.forward(self.valid_X[valid_tc - self.eval_batch_size:valid_tc])[-1]
                    valid_loss = self.loss(valid_pred, self.valid_Y[valid_tc - self.eval_batch_size:valid_tc])
                    valid_losses.append(valid_loss)
        self.end_time = time.time()

        self.train_losses = train_losses
        self.valid_losses = valid_losses

    def get_results(self, cm_norm='true'):
        loss = np.sum(np.subtract(self.Y, self.forward(self.X)[-1]) ** 2) / self.train_len
        val_loss = 0
        if self.set_valid:
            val_loss = np.sum(np.subtract(self.valid_Y, self.forward(self.valid_X)[-1]) ** 2) / self.valid_len

        accu = 0
        Y_pred = []
        Y_true = []
        for i in tqdm(range(self.train_len), ncols=100, desc='calculating train accu'):
            predicted = self.predict(self.X[i])
            expected = np.nanargmax(self.Y[i])
            Y_pred.append(predicted)
            Y_true.append(expected)
            if predicted == expected:
                accu += 1
        accu /= self.train_len
        val_accu = 0
        Y_valid_pred = []
        Y_valid_true = []
        if self.set_valid:
            for i in tqdm(range(self.valid_len), ncols=100, desc='calculating val accu'):
                predicted = self.predict(self.valid_X[i])
                expected = np.nanargmax(self.valid_Y[i])
                Y_valid_pred.append(predicted)
                Y_valid_true.append(expected)
                if predicted == expected:
                    val_accu += 1
            val_accu /= self.valid_len
        cm_train = confusion_matrix(Y_true, Y_pred, normalize=cm_norm)
        cm_valid = confusion_matrix(Y_valid_true, Y_valid_pred, normalize=cm_norm)

        elapsed_time = self.end_time - self.start_time

        results = {
            'loss': loss,
            'validation loss': val_loss,
            'accuracy': accu,
            'validation accuracy': val_accu,
            'elapsed time': elapsed_time,
            'confusion matrix train': cm_train,
            'confusion matrix validation': cm_valid,
            'train logged': range(0, self.max_iter, self.eval_interval),
            'validation logged': range(0, self.max_iter, self.eval_interval),
            'train losses': self.train_losses,
            'validation losses': self.valid_losses
        }

        return results

    def configure_reporting(self, loss_reporting=False, eval_batch_size="auto", eval_interval=10):
        self.loss_reporting = loss_reporting
        if eval_batch_size == 'auto':
            self.eval_batch_size = 'auto'
        elif (not isinstance(eval_batch_size, int)) or eval_batch_size <= 0:
            raise ValueError(f"'{eval_batch_size}' is an invalid evaluation batch size")
        else:
            self.eval_batch_size = eval_batch_size
        self.eval_interval = eval_interval
