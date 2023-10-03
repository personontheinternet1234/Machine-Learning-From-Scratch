import numpy as np


class Graph:
    def __init__(self):
        self.hidden_layers = "ERROR"

        # one for inputs, one for outputs
        self.layers_activations = [

        ]
        # don't need one for inputs
        self.layers_weights = [

        ]
        # one for outputs
        self.layers_bias = [

        ]
mygraph = Graph()
def create_graph(graph, hidden_layers):
    graph.hidden_layers = hidden_layers
create_graph(mygraph, 2)


# Creating two NumPy arrays
array1 = np.array([[1], [2], [3]])
array2 = np.array([[4], [5], [6]])

# Element-wise addition of two arrays
result = np.add(array1, array2)

hlayers = 3  # Replace 5 with the number of times you want to iterate

for layer in range(hlayers, -1, -1):  # start at last hidden layer,
    print(layer)

for layer in range(1):  # start at last hidden layer,
    print(layer)