r"""
The GardenPy package contains machine-learning tools.
These machine-learning tools are coded from scratch or PyTorch.
Additionally, GardenPy includes machine-learning models coded from these tools.
Each machine-learning model has two counterparts with similar or the same arguments.
One model is coded entirely from Scratch, only utilizing NumPy and Pandas.

Refer to 'todo' for in-depth documentation on this package.
"""

from .utils import (
    Tensor,
    nabla,
    chain,
    Initializers,
    Activators,
    Losses,
    Optimizers
)

from .models import (
    DNN,
    DNNTorch,
    CNN,
    CNNTorch
)

__all__ = [
    'Tensor',
    'nabla',
    'chain',
    'Initializers',
    'Activators',
    'Losses',
    'Optimizers',
    'DNN',
    'DNNTorch',
    'CNN',
    'CNNTorch'
]

__version__ = '0.9.0'

__authors__ = [
    'Christian SW Host-Madsen',
    'Mason Morales',
    'Isaac P Verbrugge',
    'Derek Yee'
]
