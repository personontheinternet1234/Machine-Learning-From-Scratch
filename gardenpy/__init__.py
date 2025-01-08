r"""
The GardenPy package contains machine-learning tools.
These machine-learning tools are coded from scratch or PyTorch.
Additionally, GardenPy includes machine-learning models coded from these tools.
Each machine-learning model has two counterparts with similar or the same arguments.
One model is coded entirely from Scratch, only utilizing NumPy and Pandas.

Refer to 'todo' for in-depth documentation on this package.
"""

from .functional import (
    Tensor,
    tensor,
    nabla,
    chain,
    Initializers,
    Activators,
    Losses,
    Optimizers
)
from .utils import progress

__all__ = [
    'Tensor',
    'tensor',
    'nabla',
    'chain',
    'Initializers',
    'Activators',
    'Losses',
    'Optimizers',
    'progress'
]

__version__ = '0.0.8'

__authors__ = [
    'Christian SW Host-Madsen',
    'Mason YY Morales',
    'Isaac P Verbrugge',
    'Derek S Yee'
]

__artists__ = [
    'Kamalau Kimata'
]
