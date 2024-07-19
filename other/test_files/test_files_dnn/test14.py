import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

training_data = datasets.MNIST(
    root="processed_data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="processed_data",
    train=False,
    download=True,
    transform=ToTensor()
)

print(training_data)
print('----------')
print(test_data)
