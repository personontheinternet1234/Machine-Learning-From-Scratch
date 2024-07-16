"""
Data processing for the feedforward neural network for MNIST
"""

import os

import numpy as np
import pandas as pd

# from gardenpy.utils.data_utils import (
#     format_data_raw as dr
# )

from colorama import Fore, Style
from tqdm import tqdm

from torchvision import datasets

color = f'{Fore.GREEN}{{l_bar}}{{bar}}{{r_bar}}{Style.RESET_ALL}'

# file locations
save_root = os.path.join(
    os.path.dirname(__file__), 'processed_data'
)
data_root = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__)
                )
            )
        )
    ),
    'data',
    'mnist',
    'data'
)


def process_data(save=True, values_file=os.path.join(save_root, 'values.csv'), labels_file=os.path.join(save_root, 'labels.csv')):
    # load dataset
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True)
    raw_values = np.append(train_dataset.data.numpy(), test_dataset.data.numpy(), axis=0)
    raw_labels = np.append(train_dataset.targets.numpy(), test_dataset.targets.numpy(), axis=0)

    # process data
    values = []
    labels = []
    for i in tqdm(range(len(raw_values)), ncols=100, desc='processing', bar_format=color):
        values.append(np.array(np.divide(raw_values[i].flatten().tolist(), 255)))
        out = np.zeros(10)
        out[raw_labels[i]] = 1
        out = np.array(out)
        labels.append(out)

    # create columns
    value_columns = list(range(len(values[0])))
    label_columns = list(range(len(labels[0])))

    # make list pandas dataframe
    values = pd.DataFrame(np.array(values), columns=value_columns)
    labels = pd.DataFrame(np.array(labels), columns=label_columns)

    if save:
        # save processed data
        print('saving...')
        values.to_csv(values_file, index=False)
        labels.to_csv(labels_file, index=False)
    # return processed data
    # values, labels = dr(values), dr(labels)
    # return values, labels


if __name__ == '__main__':
    process_data()
