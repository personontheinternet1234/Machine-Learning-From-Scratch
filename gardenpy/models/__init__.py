r"""
'models' includes pre-built machine-learning models for GardenPy.

Refer to 'todo' for in-depth documentation on all models.
"""

# this stuff is pretty outdated and should probably just be rewritten

from .dnn import DNN
from .dnn_torch import DNNTorch
from .cnn import CNN
from .cnn_torch import CNNTorch
from .utils import Evaluators, Savers

__all__ = [
    'DNN',
    'DNNTorch',
    'CNN',
    'CNNTorch',
    'Evaluators',
    'Savers'
]
