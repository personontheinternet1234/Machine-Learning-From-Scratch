"""
Feedforward neural network from scratch
"""

import random
import time

import numpy as np
import pandas as pd

from garden.utils.functional import (
    xavier,
    ssr,  # todo: move this to cost
    softmax,  # todo: move this to activations
    activations,
    d_activations,
    cost,
    d_cost,
)
from garden.metrics.metrics import (
    cm
)
from garden.utils.data_utils import (
    shuffle as mix
)

from colorama import Fore, Style
from tqdm import tqdm


class FNN:
    """
    Feedforward neural network
    """
    def __init__(self, weights=None, biases=None, layer_sizes='auto', activation='relu', beta=0.1, status_bars=True):
        """ model variables """
        # model version
        self.version = '1.9.0'

        # model hyperparameters
        if layer_sizes == 'auto':
            self.layer_sizes = [784, 100, 10]
        elif not isinstance(layer_sizes, list):
            raise ValueError(f"'{layer_sizes}' is not a list")
        elif len(layer_sizes) <= 2:
            raise ValueError(f"'{layer_sizes}' does not have the minimum amount of layer sizes (2)")
        elif not all(isinstance(i, int) for i in layer_sizes):
            raise ValueError(f"'{layer_sizes}' must only include integers")
        else:
            self.layer_sizes = layer_sizes
        self.weights, self.biases = self._set_parameters(weights, biases)
        self.activation = self._get_activation(activation, beta)

        # training hyperparameters
        # todo: change some of these names
        self.solver = None
        self.batch_size = None
        self.lr = None
        self.max_iter = None
        self.alpha = None

        # dataset parameters
        self.x = None
        self.y = None
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
        self.valid_batch_size = None
        self.train_losses = None
        self.valid_losses = None
        self.elapsed_time = None

        # status bar settings
        self.status = not status_bars
        self.color = f'{Fore.GREEN}{{l_bar}}{{bar}}{{r_bar}}{Style.RESET_ALL}'

    def _set_parameters(self, weights, biases):
        """ set model parameters """
        if weights is None:
            # generate weights
            weights = []
            for i in range(len(self.layer_sizes) - 1):
                length = int(self.layer_sizes[i])
                width = int(self.layer_sizes[i + 1])
                weights.append(xavier(length, width))
        else:
            # load weights
            weights = weights

        if biases is None:
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
    def _get_activation(name, beta=0.1):
        """ set model activation function """
        return {
            'function': activations(name, beta),
            'derivative': d_activations(name, beta)
        }

    @staticmethod
    def _get_loss(name):
        return {
            'function': cost(name),
            'derivative': d_cost(name)
        }

    def _get_solver(self, name):
        """ set model solving method """
        # todo: this math is wrong and needs to be fixed
        if name == 'mini-batch':
            # mini-batch gradient descent
            def update(nodes, y):
                # instantiate gradients list
                d_weights = []
                d_biases = []

                # calculate gradients
                d_b = -2 * (y - nodes[-1])
                d_biases.insert(0, d_b)
                for layer in range(-1, -len(nodes) + 1, -1):
                    d_w = np.reshape(nodes[layer - 1], (self.batch_size, self.layer_sizes[layer - 1], 1)) * d_b
                    d_weights.insert(0, d_w)
                    d_b = np.reshape(np.array([np.sum(self.weights[layer] * d_b, axis=2)]), (self.batch_size, 1, self.layer_sizes[layer - 1]))
                    d_biases.insert(0, d_b)
                d_w = np.reshape(nodes[0], (self.batch_size, self.layer_sizes[0], 1)) * d_b
                d_weights.insert(0, d_w)

                # optimize parameters
                for layer in range(len(nodes) - 1):
                    self.weights[layer] -= self.lr * np.sum((d_weights[layer] + (self.alpha / self.batch_size) * self.weights[layer]), axis=0) / self.batch_size
                    self.biases[layer] -= self.lr * np.sum(d_biases[layer], axis=0) / self.batch_size

            # return solver
            return update
        elif name == 'sgd':
            # stochastic gradient descent
            def update(nodes, y):
                # instantiate gradients list
                d_weights = []
                d_biases = []

                # calculate gradients
                d_b = -2 * (y - nodes[-1])
                d_biases.insert(0, d_b)
                for layer in range(-1, -len(nodes) + 1, -1):
                    d_w = nodes[layer - 1].T * d_b
                    d_weights.insert(0, d_w)
                    d_b = np.array([np.sum(self.weights[layer] * d_b, axis=1)])
                    d_biases.insert(0, d_b)
                d_w = nodes[0].T * d_b
                d_weights.insert(0, d_w)

                # optimize parameters
                for layer in range(len(nodes) - 1):
                    self.weights[layer] -= self.lr * (
                                d_weights[layer] + (self.alpha / self.train_len) * self.weights[layer])
                    self.biases[layer] -= self.lr * d_biases[layer]

            # return solver
            return update
        else:
            # invalid solving method
            raise ValueError(f"'{name}' is an invalid solving method")

    def forward(self, inputs):
        """ pass inputs through the model """
        nodes = [inputs]
        for layer in range(len(self.layer_sizes) - 1):
            node_layer = self.activation['function'](np.matmul(nodes[-1], self.weights[layer]) + self.biases[layer])
            nodes.append(node_layer)
        return nodes

    def fit(self, x, y, solver='mini-batch', batch_size='auto', learning_rate=0.001, max_iter=20000, alpha=0.0001, shuffle=False):
        """ optimize model """
        # set training hyperparameters
        self.x, self.y = mix(np.array(x), np.array(y))
        self.solver = self._get_solver(solver)
        self.train_len = len(self.x)
        self.lr = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha
        if batch_size == 'auto':
            self.batch_size = min(20, self.train_len)
        elif batch_size == 'all':
            self.batch_size = self.train_len
        elif (not isinstance(batch_size, int)) or batch_size <= 1:
            raise ValueError(f"'{batch_size}' is an invalid batch size")
        else:
            self.batch_size = batch_size
        if self.eval_batch_size == 'auto' and self.set_valid:
            self.eval_batch_size = min(20, self.valid_len)
            self.valid_batch_size = self.eval_batch_size
        elif self.eval_batch_size == 'all' and self.set_valid:
            self.eval_batch_size = self.train_len
            self.valid_batch_size = self.valid_len
        elif self.set_valid:
            self.valid_batch_size = self.eval_batch_size

        # set loss lists
        self.train_losses = []
        self.valid_losses = []
        # start timer
        start_time = time.time()

        for batch in tqdm(range(max_iter), ncols=100, desc='fitting', disable=self.status, bar_format=self.color):
            # training loop
            # shuffle data
            if shuffle:
                self.x, self.y = mix(self.x, self.y)
            # set input batches
            if solver == 'mini-batch':
                tc = random.randint(self.batch_size, self.train_len)
                in_n = self.x[tc - self.batch_size:tc]
                out_n = self.y[tc - self.batch_size:tc]
            elif solver == 'sgd':
                tc = random.randint(0, self.train_len - 1)
                in_n = x[tc]
                out_n = y[tc]
            else:
                raise ValueError('there was a change to an invalid solver')

            # optimize network
            nodes = self.forward(in_n)
            self.solver(nodes, out_n)

            # loss calculation
            if self.loss_reporting and batch % self.eval_interval == 0:
                tc = random.randint(self.eval_batch_size, self.train_len)
                train_pred = self.forward(self.x[tc - self.eval_batch_size:tc])[-1]
                train_loss = ssr(train_pred, self.y[tc - self.eval_batch_size:tc]) / self.eval_batch_size
                self.train_losses.append(train_loss)
                if self.set_valid:
                    valid_tc = random.randint(self.valid_batch_size, self.valid_len)
                    valid_pred = self.forward(self.valid_x[valid_tc - self.valid_batch_size:valid_tc])[-1]
                    valid_loss = ssr(valid_pred,
                                     self.valid_y[valid_tc - self.valid_batch_size:valid_tc]) / self.valid_batch_size
                    self.valid_losses.append(valid_loss)

        # calculate elapsed time
        self.elapsed_time = time.time() - start_time

    def predict(self, inputs):
        """ predict outputs based on the model """
        predicted = np.nanargmax(self.forward(inputs)[-1])
        return predicted

    def validation(self, valid_x, valid_y):
        """ set validation dataset """
        self.set_valid = True
        self.valid_x, self.valid_y = mix(valid_x, valid_y)
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
        elif eval_batch_size == 'all':
            self.eval_batch_size = 'all'
        elif (not isinstance(eval_batch_size, int)) or eval_batch_size <= 0:
            raise ValueError(f"'{eval_batch_size}' is an invalid evaluation batch size")
        else:
            self.eval_batch_size = eval_batch_size

    def get_results(self, cm_norm=True):
        """ return network results to the user """
        # calculate training dataset loss using MSE
        mean_train_loss = ssr(self.y, self.forward(self.x)[-1]) / self.train_len
        # calculate validation dataset loss
        mean_val_loss = None
        if self.set_valid:
            mean_val_loss = ssr(self.valid_y, self.forward(self.valid_x)[-1]) / self.valid_len

        # instantiate training dataset dictionary
        train_outcomes = {
            'label': [],
            'predicted': [],
            'accurate': [],
            'loss': [],
            'probability': []
        }
        for i in tqdm(range(self.train_len), ncols=100, desc='train', disable=self.status, bar_format=self.color):
            # calculate training dataset results
            expected = self.y[i]
            predicted = self.forward(self.x[i])[-1]
            train_outcomes['label'].append(np.nanargmax(expected))
            train_outcomes['predicted'].append(np.nanargmax(predicted))
            train_outcomes['accurate'].append(np.nanargmax(expected) == np.nanargmax(predicted))
            train_outcomes['loss'].append(ssr(expected, predicted))
            train_outcomes['probability'].append(softmax(predicted)[0])
        train_accu = np.sum(train_outcomes['accurate']) / self.train_len

        val_outcomes = None
        val_accu = None
        if self.set_valid:
            # instantiate validation dataset dictionary
            val_outcomes = {
                'label': [],
                'predicted': [],
                'accurate': [],
                'loss': [],
                'probability': []
            }
            for i in tqdm(range(self.valid_len), ncols=100, desc='valid', disable=self.status, bar_format=self.color):
                # calculate validation dataset results
                expected = self.valid_y[i]
                predicted = self.forward(self.valid_x[i])[-1]
                val_outcomes['label'].append(np.nanargmax(expected))
                val_outcomes['predicted'].append(np.nanargmax(predicted))
                val_outcomes['accurate'].append(np.nanargmax(expected) == np.nanargmax(predicted))
                val_outcomes['loss'].append(ssr(expected, predicted))
                val_outcomes['probability'].append(softmax(predicted)[0])
            val_accu = np.sum(val_outcomes['accurate']) / self.valid_len

        # generate training and validation confusion matrices
        cm_train = cm(train_outcomes['label'], train_outcomes['predicted'], normalize=cm_norm)
        cm_valid = None
        if self.set_valid:
            cm_valid = cm(val_outcomes['label'], val_outcomes['predicted'], normalize=cm_norm)

        # reformat outcomes to pandas dataframe
        train_outcomes = pd.DataFrame(train_outcomes)
        val_outcomes = pd.DataFrame(val_outcomes)

        # reformat logged losses to pandas dataframe
        logged_losses = {
            'logged points': range(0, self.max_iter, self.eval_interval),
            'training losses': self.train_losses
        }
        if self.set_valid:
            logged_losses['validation losses'] = self.valid_losses
        logged_losses = pd.DataFrame(logged_losses)

        # reformat single results into singular pandas dataframe
        final_results = {
            'label': ['mean training loss', 'training accuracy'],
            'result': [mean_train_loss, train_accu]
        }
        if self.set_valid:
            final_results['label'].append('mean validation loss')
            final_results['label'].append('validation accuracy')
            final_results['result'].append(mean_val_loss)
            final_results['result'].append(val_accu)
        final_results['label'].append('elapsed training time')
        final_results['result'].append(self.elapsed_time)
        final_results = dict(zip(final_results['label'], final_results['result']))
        final_results = pd.Series(final_results)

        # format results dictionary
        results = {
            'final results': final_results,
            'training confusion matrix': cm_train,
            'validation confusion matrix': cm_valid,
            'logged losses': logged_losses,
            'training outcomes': train_outcomes,
            'validation outcomes': val_outcomes,
            'weights': self.weights,
            'biases': self.biases
        }

        # return results dictionary
        return results
