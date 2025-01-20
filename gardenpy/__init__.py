r"""
GardenPy is an autograd library with integrated machine learning algorithms.

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
from .models import (
    DNN,
    CNN
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
    'DNN',
    'CNN',
    'progress'
]

__version__ = '0.0.8'

__authors__ = [
    'Christian SW Host-Madsen',
    'Doyoung Kim',
    'Mason YY Morales',
    'Isaac P Verbrugge',
    'Derek S Yee'
]

__artists__ = [
    'Kamalau Kimata'
]
