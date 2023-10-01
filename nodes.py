class Graph:
    # Overall graph. Not sure when I'm going to use this. Useful mainly just to list all the nodes.

    def __init__(self, name):
        self.name = name
        self.nodes = []

        # Don't worry! Self.layers and its friends now scale for the # of layers we've got!!
        self.layers = [
            [], []
        ]
        self.layers_sums = [
            [], []
        ]
        self.layers_activations = [
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
        self.backconnections = []
        self.bias = bias

        self.connections_weights = []
        self.backconnections_gradients = []

    def new_connection(self, origin, destination, weight):
        forward_connection = Connection(origin, destination, weight)
        self.connections.append(forward_connection)

        back_connection = BackConnection(destination, origin)
        destination.backconnections.append(back_connection)

    def fix_connections_weights(self):
        # makes a list per every starting node, with every outgoing node as parts of that list
        # logically, this will end up as a matrix of n x k (we want k x n)
        # n vertically (n lists, where n is the # of coming from), k horizontally (where k is where going to)
        for connection in range(len(self.connections)):
            try:
                self.connections_weights[connection] = self.connections[connection].weight
            except:
                self.connections_weights.append(self.connections[connection].weight)



class Connection:
    # Connections for a node. Very useful. Much math will be done with connections.

    def __init__(self, origin, destination, weight):
        self.origin = origin
        self.destination = destination
        self.weight = weight

    def return_name(self):
        return str(self.origin.name) + " to " + str(self.destination.name)

class BackConnection:
    # Back Connections for backprop

    def __init__(self, origin, destination, gradient=0):
        self.origin = origin
        self.destination = destination
        self.gradient = gradient

    def return_name(self):
        return str(self.origin.name) + " to " + str(self.destination.name)

if __name__ == "__main__":
    mygraph = Graph("mygraph")
    HI = Node("hi", mygraph, 0)
    CA = Node("ca", mygraph, 1)
    AZ = Node("az", mygraph, 1)

    HI.new_connection(HI, CA, 1.0)
    HI.new_connection(HI, AZ, 1.0)