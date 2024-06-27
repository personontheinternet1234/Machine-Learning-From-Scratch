"""
processed_data utility functions
"""

import ast
import random

import numpy as np
import pandas as pd

from garden.utils.helper_functions import print_color

from colorama import Fore, Style
from tqdm import tqdm

tqdm_color = f'{Fore.GREEN}{{l_bar}}{{bar}}{{r_bar}}{Style.RESET_ALL}'


def trim(data, trim_frac=0.5):
    """ trim data based on a fraction """
    data = data[0:trim_frac * len(data)]
    return data


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


def shuffle(values, labels):
    """ shuffle values and labels """
    # zip & shuffle data
    data = list(zip(values, labels))
    random.shuffle(data)
    # unzip data
    values, labels = zip(*data)
    # return data
    return values, labels


def format_parameters(file_path, status_bars=True):
    """ format parameters from a .csv file """
    # find errors
    if not file_path.endswith('.txt'):
        raise ValueError(f'{file_path} is not a .txt file')
    # instantiate parameters list
    parameters = []
    # load parameters
    if status_bars:
        print_color('formatting parameters...')
    with open(file_path, 'r') as f:
        for line in f:
            # reformat parameters
            parameters.append(np.array(ast.literal_eval(line)))
    # return parameters
    return parameters


def format_data(file_path, status_bars=True):
    """ format data from a .csv file """
    # find errors
    if not file_path.endswith('.csv'):
        raise ValueError(f'{file_path} is not a .csv file')
    # load data
    if status_bars:
        print_color('loading dataframe...')
    arr = np.array(pd.read_csv(file_path)).tolist()
    # reformat data
    for i in tqdm(range(len(arr)), ncols=100, desc='formatting', disable=not status_bars, bar_format=tqdm_color):
        arr[i] = np.array([arr[i]])
    # return data
    return np.array(arr)


def format_data_raw(df, status_bars=True):
    """ format data from a pandas dataframe """
    arr = np.array(df).tolist()
    for i in tqdm(range(len(arr)), ncols=100, desc='formatting', disable=not status_bars, bar_format=tqdm_color):
        arr[i] = np.array([arr[i]])
    # return processed_data
    return np.array(arr)


def save_parameters(file_path, parameters, status_bars=True):
    """ save parameters into a .txt file """
    # find errors
    if not file_path.endswith('.txt'):
        raise ValueError(f'{file_path} is not a .txt file')
    # save parameters
    with open(file_path, 'w') as f:
        for layer in tqdm(range(len(parameters)), ncols=100, desc='saving', disable=not status_bars, bar_format=tqdm_color):
            f.write(str(parameters[layer].tolist()) + "\n")
    return None
