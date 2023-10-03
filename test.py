import numpy as np


class Graph:
    def __init__(self, total_layers):
        self.total_layers = total_layers

        # one for inputs, one for outputs
        self.layers_activations = [

        ]
        # don't need one for inputs
        self.layers_weights = [

        ]
        # one for outputs
        self.layers_bias = [

        ]
mygraph = Graph(0)

print((np.zeros((5, 1))))