r"""
Dense Neural Network training on MNIST.
"""

import os

from gardenpy import DNN
from gardenpy.utils import DataLoaderCSV, ansi
from gardenpy.utils.helper_functions import print_credits

print()
print_credits()

dataset = DataLoaderCSV(
    labels='labels',
    values='values',
    root=os.path.join(os.path.dirname(__file__), 'processed_data'),
    valid=0.3,
    batching=64,
    trim=False,
    shuffle=True,
    save_memory=True,
)

model = DNN(status_bars=True)
model.initialize(
    hidden_layers=[16, 16],
    thetas={
        'weights': [
            {'algorithm': 'xavier', 'gain': 1.0},
            {'algorithm': 'xavier', 'gain': 1.0},
            {'algorithm': 'xavier', 'gain': 1.0}
        ],
        'biases': [
            {'algorithm': 'zeros'},
            {'algorithm': 'zeros'},
            {'algorithm': 'zeros'}
        ]
    },
    activations=[
        {'algorithm': 'lrelu', 'beta': 0.1},
        {'algorithm': 'lrelu', 'beta': 0.1},
        {'algorithm': 'softmax'}
    ]
)
model.hyperparameters(
    loss={'algorithm': 'centropy'},
    optimizer={
        'algorithm': 'adam',
        'gamma': 1e-2,
        'lambda_d': 0.01,
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-10,
        'ams': False
    }
)
model.fit(
    data=dataset,
    parameters={
        'iterations': 10_000,
        'rate': 10
    }
)
model.eval(
    graphs={
        'boxplot': {'on': True},
        'heatmap': {'on': True, 'normalized': True, 'validation': True},
        'lgraph': {'on': True, 'accuracy': True, 'loss': True, 'validation': True},
        'rgraph': {'on': True, 'accuracy': True, 'loss': True, 'validation': True}
    }
)
weights, biases = model.final()
save_parameters = input(f"Input '{ansi['white']}{ansi['italic']}save{ansi['reset']}' to save parameters: ")
if save_parameters.lower() == 'save':
    print('not done yet :(')
else:
    print(f"{ansi['bright_black']}{ansi['italic']}Parameters not saved.{ansi['reset']}")
