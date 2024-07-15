"""
Feedforward Neural Network coded with PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

from colorama import Fore, Style
from tqdm import tqdm


class FNNTorch:
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
        self.weights, self.biases = self.set_parameters(weights, biases)
        self.activation = self._get_activation(activation, beta)

        # training hyperparameters
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

    class Core(nn.Module):
        def __init__(self):
            super(FNNTorch.Core, self).__init__()
            self.layers = []
