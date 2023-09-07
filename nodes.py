import numpy as np

class Graph:
    # Overall graph. Not sure when I'm going to use this. Useful mainly just to list all the nodes.

    def __init__(self, name):
        self.name = name
        self.nodes = []
        # I should change this to account for # of variables when I scale it
        self.layers = [
            [], []
        ]
        self.layers_activations = [
            [], []
        ]
        self.layers_bias = [
            [], []
        ]

    def add_node(self, name):
        self.nodes.append(name)
        self.layers[name.layer].append(name)


class Node:
    # Each Node. Very useful. node.activationEnergy will be changed a lot.

    def __init__(self, name, graph, layer, energy=0, bias=0):
        self.name = name
        self.graph = graph
        self.layer = layer
        self.activationEnergy = energy
        self.connections = []
        self.bias = bias

        self.connections_weights = []

    def new_connection(self, name, destination, weight):
        name = Connection(name, destination, weight)
        self.connections.append(name)

        # makes a list per every starting node, with every outgoing node as parts of that list
        # logically, this will end up as a matrix of n x k (we want k x n)
        # n vertically (n lists, where n is the # of coming from), k horizontally (where k is where going to)
        self.connections_weights.append(name.weight)


        # destination.connections.append(name)

class Connection:
    # Connections for a node. Very useful. Much math will be done with connections.

    def __init__(self, origin, destination, weight):
        self.origin = origin
        self.destination = destination
        self.weight = weight

    def return_name(self):
        return str(self.origin.name) + " to " + str(self.destination.name)

def forward(graph):  # forward pass
    new_list = []
    for node in graph.layers[0]:
        new_list += node.connections_weights
    new_list = np.array(new_list)
    return np.matmul(graph.layers_activations[0], new_list)

if __name__ == "__main__":
    mygraph = Graph("mygraph")
    HI = Node("hi", mygraph, 0)
    CA = Node("ca", mygraph, 1)
    AZ = Node("az", mygraph, 1)

    HI.new_connection(HI, CA, 1.0)
    HI.new_connection(HI, AZ, 1.0)

    print(forward(mygraph))