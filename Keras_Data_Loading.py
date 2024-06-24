"""
MNIST loading process
Authors:
    Christian SW Host-Madsen CO '25 <chost-madsen25@punahou.edu>
    Isaac Park Verbrugge CO '25 <iverbrugge25@punahou.edu>
"""

import os

import numpy as np

from colorama import Fore, Style
from tqdm import tqdm

import keras


def main():
    # locations & settings
    save_location_dir = os.path.join(os.path.dirname(__file__), 'assets', 'data')
    save_location_values_name = 'data_values_keras.csv'
    save_location_labels_name = 'data_labels_keras.csv'

    num_labels = 10
    color = f'{Fore.GREEN}{{l_bar}}{{bar}}{{r_bar}}{Style.RESET_ALL}'

    # load data from mnist
    (train_values, train_labels), (test_values, test_labels) = keras.datasets.mnist.load_data()
    # combine training and testing labels
    import_data_values = np.append(train_values, test_values, axis=0)
    import_data_labels = np.append(train_labels, test_labels, axis=0)

    # reformat data from mnist and add labels
    data_values = []
    data_labels = []
    for i in tqdm(range(len(import_data_values)), ncols=100, desc='formatting', bar_format=color):
        data_values.append(np.array([np.divide(import_data_values[i].flatten().tolist(), 255)]))
        node_values = np.zeros(num_labels)
        node_values[import_data_labels[i]] = 1
        node_values = np.array([node_values])
        data_labels.append(node_values)

    # generate data names
    name_values = []
    for i in tqdm(range(len(import_data_values[0]) ** 2), ncols=100, desc='values', bar_format=color):
        name_values.append(i)
    name_labels = []
    for i in tqdm(range(num_labels), ncols=100, desc='labels', bar_format=color):
        name_labels.append(i)

    # save keras data
    with open(os.path.join(save_location_dir, save_location_values_name), 'w') as f:
        f.write(str(name_values).strip('[').strip(']') + '\n')
        for array in tqdm(range(len(data_values)), ncols=100, desc='saving', bar_format=color):
            f.write(str(data_values[array].tolist()[0]).strip('[').strip(']') + '\n')
    with open(os.path.join(save_location_dir, save_location_labels_name), 'w') as f:
        f.write(str(name_labels).strip('[').strip(']') + '\n')
        for array in tqdm(range(len(data_labels)), ncols=100, desc='saving', bar_format=color):
            f.write(str(data_labels[array].tolist()[0]).strip('[').strip(']') + '\n')


if __name__ == '__main__':
    main()
