"""
Data generator for MNIST data
"""

import os
import shutil

from torchvision import datasets
from torchvision.transforms import ToTensor

# set root and required files
root = os.path.join(os.path.dirname(__file__), 'data')
required_files = [
    't10k-images-idx3-ubyte',
    't10k-labels-idx1-ubyte',
    'train-images-idx3-ubyte',
    't10k-images-idx3-ubyte.gz',
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    'train-labels-idx1-ubyte',
    't10k-labels-idx1-ubyte.gz'
]

# list existing files
existing_files = []
if os.path.exists(os.path.join(root, 'KERAS', 'raw')):
    for file in os.listdir(os.path.join(root, 'KERAS', 'raw')):
        existing_files.append(file)

# check if data is valid
if required_files != existing_files:
    # clear root
    shutil.rmtree(root)
    os.makedirs(root)

    # load raw data
    training_data = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=ToTensor()
    )

    # # move files straight to data
    # for file in os.listdir(os.path.join(root, 'MNIST', 'raw')):
    #     shutil.move(os.path.join(root, 'MNIST', 'raw', file), root)
    #
    # # clear generated MNIST folder
    # shutil.rmtree(os.path.join(root, 'MNIST'))
