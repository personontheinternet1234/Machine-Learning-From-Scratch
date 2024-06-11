"""
Authors:
    Isaac Park Verbrugge CO '25 <iverbrugge25@punahou.edu>
    Christian SW Host-Madsen CO '25 <chost-madsen25@punahou.edu>
"""

import math
import random
import time

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from colorama import Fore, Style


class NeuralNetwork:
    """
    Fully connected neural network (FNN).
    """

    def __init__(self, weights='none', biases='none', layer_sizes='auto', activation='relu', status_bars=True):
        """ model variables """
        # model version
        self.version = '1.7.0'
        # model hyperparameters
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
        # training hyperparameters
        self.solver = None
        self.batch_size = None
        self.learning_rate = None
        self.max_iter = None
        self.alpha = None
        self.momentum = None
        # dataset parameters
        self.x = None
        self.y = None
        self.array_x = None
        self.array_y = None
        self.train_len = None
        self.set_valid = False
        self.valid_x = None
        self.valid_y = None
        self.array_valid_x = None
        self.array_valid_y = None
        self.valid_len = None
        # output settings
        self.loss_reporting = False
        self.eval_interval = None
        self.eval_batch_size = None
        self.train_losses = None
        self.valid_losses = None
        self.elapsed_time = None
        # status bar settings
        self.status = not status_bars
        self.color = f'{Fore.GREEN}{{l_bar}}{{bar}}{{r_bar}}{Style.RESET_ALL}'

    def set_parameters(self, weights, biases):
        """ set model parameters """
        if weights == 'none':
            # generate weights
            weights = []
            for i in range(len(self.layer_sizes) - 1):
                length = int(self.layer_sizes[i])
                width = int(self.layer_sizes[i + 1])
                weights.append(np.random.randn(length, width) * math.sqrt(2 / length))
        else:
            # load weights
            weights = weights

        if biases == 'none':
            # generate biases
            biases = []
            for i in range(len(self.layer_sizes) - 1):
                length = 1
                width = self.layer_sizes[i + 1]
                biases.append(np.zeros((length, width)))
        else:
            # load biases
            biases = biases

        # return weights and biases
        return weights, biases

    @staticmethod
    def _get_activation(name):
        """ set model activation function """
        if name == 'sigmoid':
            # sigmoid activation function
            return {
                'forward': lambda x: 1 / (1 + np.exp(-x)),
                'derivative': lambda x: x * (1 - x)
            }
        elif name == 'tanh':
            # hyperbolic tan activation function
            return {
                'forward': lambda x: np.tanh(x),
                'derivative': lambda x: 1 - np.tanh(x) ** 2
            }
        elif name == 'relu':
            # rectified linear unit activation function
            return {
                'forward': lambda x: np.maximum(0, x),
                'derivative': lambda x: np.where(x > 0, 1, 0)
            }
        elif name == 'leaky relu':
            # leaky rectified linear unit activation function
            return {
                'forward': lambda x: np.maximum(0.1 * x, x),
                'derivative': lambda x: np.where(x > 0, 1, 0.1)
            }
        else:
            # invalid activation function
            raise ValueError(f"'{name}' is an invalid activation function")

    def _get_solver(self, name):
        """ set model solving method """
        if name == 'sgd':
            # stochastic gradient descent
            def update(nodes, y, weights, biases, learning_rate, alpha, momentum):
                # instantiate gradients list
                d_weights = []
                d_biases = []

                # calculate gradients
                d_b = -2 * (y - nodes[-1])
                d_biases.insert(0, d_b)
                for layer in range(-1, -len(nodes) + 1, -1):
                    d_w = nodes[layer - 1].T * d_b
                    d_weights.insert(0, d_w)
                    d_b = np.array([np.sum(weights[layer] * d_b, axis=1)])
                    d_biases.insert(0, d_b)
                d_w = nodes[0].T * d_b
                d_weights.insert(0, d_w)

                # optimize parameters
                for layer in range(len(nodes) - 1):
                    weights[layer] -= learning_rate * (d_weights[layer] + (alpha / self.train_len) * weights[layer])
                    biases[layer] -= learning_rate * d_biases[layer]

                # return optimized parameters
                return weights, biases

            return {'update': update}
        elif name == 'mini-batch':
            # mini-batch gradient descent
            def update(nodes, y, weights, biases, learning_rate, alpha, momentum):
                # instantiate gradients list
                d_weights = []
                d_biases = []

                # calculate gradients
                d_b = -2 * (y - nodes[-1])
                d_biases.insert(0, d_b)
                for layer in range(-1, -len(nodes) + 1, -1):
                    d_w = np.reshape(nodes[layer - 1], (self.batch_size, self.layer_sizes[layer - 1], 1)) * d_b
                    d_weights.insert(0, d_w)
                    d_b = np.reshape(np.array([np.sum(weights[layer] * d_b, axis=2)]), (self.batch_size, 1, self.layer_sizes[layer - 1]))
                    d_biases.insert(0, d_b)
                d_w = np.reshape(nodes[0], (self.batch_size, self.layer_sizes[0], 1)) * d_b
                d_weights.insert(0, d_w)

                # optimize parameters
                for layer in range(len(nodes) - 1):
                    weights[layer] -= learning_rate * np.sum((d_weights[layer] + (alpha / self.batch_size) * weights[layer]), axis=0) / self.batch_size
                    biases[layer] -= learning_rate * np.sum(d_biases[layer], axis=0) / self.batch_size

                # return optimized parameters
                return weights, biases

            return {'update': update}
        else:
            # invalid solving method
            raise ValueError(f"'{name}' is an invalid solving method")

    @staticmethod
    def ssr(expected, predicted):
        """ calculate loss using the sum of the squared residuals """
        loss = np.sum(np.subtract(expected, predicted) ** 2)
        return loss

    @staticmethod
    def softmax(values):
        """ calculate outcome probabilities using softmax """
        output = np.exp(values) / np.sum(np.exp(values))
        return output

    def forward(self, inputs):
        """ pass inputs through the model """
        nodes = [inputs]
        for layer in range(len(self.layer_sizes) - 1):
            node_layer = self.activation['forward'](np.matmul(nodes[-1], self.weights[layer]) + self.biases[layer])
            nodes.append(node_layer)
        return nodes

    def fit(self, x, y, solver='mini-batch', alpha=0.0001, batch_size='auto', learning_rate=0.001, max_iter=20000, momentum=0.9):
        """ optimize model """
        # set training hyperparameters
        self.x = x
        self.y = y
        self.solver = self._get_solver(solver)
        self.array_x = np.array(x)
        self.array_y = np.array(y)
        self.train_len = len(self.x)
        self.alpha = alpha
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        if batch_size == 'auto':
            self.batch_size = min(20, len(self.x))
        elif (not isinstance(batch_size, int)) or batch_size <= 1:
            raise ValueError(f"'{batch_size}' is an invalid batch size")
        else:
            self.batch_size = batch_size
        if self.eval_batch_size == 'auto' and self.set_valid:
            self.eval_batch_size = min(20, len(self.valid_x))
        elif self.eval_batch_size == 'auto':
            self.eval_batch_size = min(20, len(self.x))

        # reset loss lists
        self.train_losses = []
        self.valid_losses = []
        # start timer
        start_time = time.time()

        for batch in tqdm(range(max_iter), ncols=100, desc='fitting', disable=self.status, bar_format=self.color):
            # training loop
            # set input batches
            if solver == 'mini-batch':
                tc = random.randint(self.batch_size, self.train_len)
                in_n = self.array_x[tc - self.batch_size:tc]
                out_n = self.array_y[tc - self.batch_size:tc]
            elif solver == 'sgd':
                tc = random.randint(0, self.train_len - 1)
                in_n = x[tc]
                out_n = y[tc]
            else:
                raise ValueError('there was a change to an invalid solver')

            # optimize network
            nodes = self.forward(in_n)
            self.weights, self.biases = self.solver['update'](nodes, out_n, self.weights, self.biases, learning_rate, alpha, momentum)

            # loss calculation
            if self.loss_reporting and batch % self.eval_interval == 0:
                tc = random.randint(self.eval_batch_size, self.train_len)
                train_pred = self.forward(self.x[tc - self.eval_batch_size:tc])[-1]
                train_loss = self.ssr(train_pred, self.y[tc - self.eval_batch_size:tc])
                self.train_losses.append(train_loss)
                if self.set_valid:
                    valid_tc = random.randint(self.eval_batch_size, self.valid_len)
                    valid_pred = self.forward(self.valid_x[valid_tc - self.eval_batch_size:valid_tc])[-1]
                    valid_loss = self.ssr(valid_pred, self.valid_y[valid_tc - self.eval_batch_size:valid_tc])
                    self.valid_losses.append(valid_loss)

        # calculate elapsed time
        end_time = time.time()
        self.elapsed_time = end_time - start_time

    def predict(self, inputs):
        """ predict outputs based on the model """
        predicted = np.nanargmax(self.forward(inputs)[-1])
        return predicted

    def validation(self, valid_x, valid_y):
        """ set validation dataset """
        self.set_valid = True
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.array_valid_x = np.array(valid_x)
        self.array_valid_y = np.array(valid_y)
        self.valid_len = len(self.valid_x)

    def configure_reporting(self, loss_reporting=False, eval_batch_size="auto", eval_interval=10):
        """ configure how loss reporting is done """
        # determine if loss reporting will be done
        self.loss_reporting = loss_reporting
        # set evaluation interval for loss reporting
        self.eval_interval = eval_interval
        # set evaluation batch sizes for loss reporting
        if eval_batch_size == 'auto':
            self.eval_batch_size = 'auto'
        elif (not isinstance(eval_batch_size, int)) or eval_batch_size <= 0:
            raise ValueError(f"'{eval_batch_size}' is an invalid evaluation batch size")
        else:
            self.eval_batch_size = eval_batch_size

    # todo
    def get_results(self, cm_norm='true'):
        """ return network results to the user """
        # calculate training dataset loss
        mean_loss = self.ssr(self.y, self.forward(self.x)[-1]) / self.train_len
        # calculate validation dataset loss
        mean_val_loss = 0
        if self.set_valid:
            mean_val_loss = self.ssr(self.valid_y, self.forward(self.valid_x)[-1]) / self.valid_len

        train_outcomes = {
            'label': [],
            'correct': [],
            'losses': [],
            'probabilities': []
        }
        # calculate training dataset loss
        accu = 0
        y_pred = []
        y_true = []
        for i in tqdm(range(self.train_len), ncols=100, desc='train results', disable=self.status, bar_format=self.color):
            predicted = self.predict(self.x[i])
            expected = np.nanargmax(self.y[i])
            train_outcomes['label'].append(expected)
            train_outcomes['correct'].append(predicted == expected)
            y_pred.append(predicted)
            y_true.append(expected)
            if predicted == expected:
                accu += 1
        accu /= self.train_len

        # calculate validation dataset loss
        val_accu = 0
        y_valid_pred = []
        y_valid_true = []
        if self.set_valid:
            for i in tqdm(range(self.valid_len), ncols=100, desc='val results', disable=self.status, bar_format=self.color):
                predicted = self.predict(self.valid_x[i])
                expected = np.nanargmax(self.valid_y[i])
                y_valid_pred.append(predicted)
                y_valid_true.append(expected)
                if predicted == expected:
                    val_accu += 1
            val_accu /= self.valid_len

        # generate training and validation confusion matrices
        cm_train = confusion_matrix(y_true, y_pred, normalize=cm_norm)
        cm_valid = confusion_matrix(y_valid_true, y_valid_pred, normalize=cm_norm)

        # combine results into a dictionary
        results = {
            'mean loss': mean_loss,
            'mean validation loss': mean_val_loss,
            'accuracy': accu,
            'validation accuracy': val_accu,
            'elapsed training time': self.elapsed_time,
            'training confusion matrix': cm_train,
            'validation confusion matrix': cm_valid,
            'train losses': self.train_losses,
            'validation losses': self.valid_losses,
            'logged points': range(0, self.max_iter, self.eval_interval)
        }

        # return results dictionary
        return results

    def print_version(self):
        """ print neural network version """
        print(f'{self.version}')
