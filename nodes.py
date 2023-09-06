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

    def new_connection(self, name, destination, weight):
        name = Connection(name, destination, weight)
        self.connections.append(name)

        destination.connections.append(name)

class Connection:
    # Connections for a node. Very useful. Much math will be done with connections.

    def __init__(self, origin, destination, weight):
        self.origin = origin
        self.destination = destination
        self.weight = weight

    def return_name(self):
        return str(self.origin.name + " to " + self.destination.name)