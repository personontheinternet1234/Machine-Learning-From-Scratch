"""
File formatting functions
Authors:
    Christian SW Host-Madsen CO '25 <chost-madsen25@punahou.edu>
    Isaac Park Verbrugge CO '25 <iverbrugge25@punahou.edu>
"""

import ast

import numpy as np
import pandas as pd

from Garden.Functions.Metrics import print_color

from colorama import Fore, Style
from tqdm import tqdm

tqdm_color = f'{Fore.GREEN}{{l_bar}}{{bar}}{{r_bar}}{Style.RESET_ALL}'


def format_parameters(file_path, status_bars=True):
    """
    format parameters from a .txt file

    Weights txt should look like:
    [[a,b,c...], [d,e,f...]...]
    [[g,h...], [i,j...], [k,l...]...]

    eg:
    [[-0.8771341087742135, 1.578797175316903, -2.421074385347355], [1.0226131651189596, -1.1367997111676496, 1.1616987542888868]]
    [[0.8061576470339566, -0.29437974521184074], [0.909628881514868, -0.9623878282632817], [0.20515100022079089, -0.8415196464701109]]

    Biases txt should look like:
    [[a,b,c...]]
    [[g,h...]]

    eg:
    [[0.033332371717526704, -0.512611285317706, -0.41990002117387404]]
    [[0.06786937893477499, 0.8402237807256552]]
    """

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
    df = np.array(pd.read_csv(file_path)).tolist()
    # reformat data
    for i in tqdm(range(len(df)), ncols=100, desc='formatting', disable=not status_bars, bar_format=tqdm_color):
        df[i] = np.array([df[i]])
    # return data
    return df


def save_parameters(file_path, parameters, status_bars=True):
    """ save parameters into a .txt file """
    # find errors
    if not file_path.endswith('.csv'):
        raise ValueError(f'{file_path} is not a .csv file')
    # save parameters
    with open(file_path, 'w') as f:
        for layer in tqdm(range(len(parameters)), ncols=100, desc='saving', disable=not status_bars, bar_format=tqdm_color):
            f.write(str(parameters[layer].tolist()) + "\n")
    return None
