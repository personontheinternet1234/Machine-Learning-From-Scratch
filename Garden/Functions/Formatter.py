"""
File formatting functions
Authors:
    Christian SW Host-Madsen CO '25 <chost-madsen25@punahou.edu>
    Isaac Park Verbrugge CO '25 <iverbrugge25@punahou.edu>
"""


import ast

import numpy as np
import pandas as pd

from colorama import Fore, Style
from tqdm import tqdm

tqdm_color = f'{Fore.GREEN}{{l_bar}}{{bar}}{{r_bar}}{Style.RESET_ALL}'


def format_parameters(file_path):
    """ format parameters from a .txt file"""
    # find errors
    if file_path[-4:] != '.txt':
        raise ValueError('file is not a .txt file')
    # instantiate parameters list
    parameters = []
    # load parameters
    with open(file_path, 'r') as f:
        for line in f:
            # reformat parameters
            parameters.append(np.array(ast.literal_eval(line)))
    # return parameters
    return parameters


def format_data(file_path, status_bars=True):
    # find errors
    if file_path[-4:] != '.csv':
        raise ValueError('file is not a .csv file')
    """ format data from a .csv file """
    # load data
    df = np.array(pd.read_csv(file_path)).tolist()
    # reformat data
    for i in tqdm(range(len(df)), ncols=100, desc='formatting', disable=not status_bars, bar_format=tqdm_color):
        df[i] = np.array([df[i]])
    # return data
    return df
