r"""
'models' includes pre-built machine-learning models for GardenPy.

Refer to 'todo' for in-depth documentation on all models.
"""

from .dnn import DNN

from .dnn_torch import DNNTorch

from .cnn import CNN

from .cnn_torch import CNNTorch

__all__ = [
    'DNN',
    'DNNTorch',
    'CNN',
    'CNNTorch'
]
