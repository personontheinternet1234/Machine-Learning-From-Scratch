class Graph:
    # Overall graph. Not sure when I'm going to use this. Useful mainly just to list all the nodes.

    def __init__(self, name):
        self.name = name
        self.nodes = []

    def add_node(self, name):
        self.nodes.append(name)


class Node:
    # Each Node. Very useful. node.activationEnergy will be changed a lot.

    def __init__(self, name, energy=0):
        self.name = name
        self.activationEnergy = energy
        self.connections = []

    def new_connection(self, name, destination, weight, bias):
        name = Connection(name, destination, weight, bias)
        self.connections.append(name)


class Connection:
    # Connections for a node. Very useful. Much math will be done with connections.

    def __init__(self, origin, destination, weight, bias):
        self.origin = origin
        self.destination = destination
        self.weight = weight
        self.bias = bias

    def return_name(self):
        return str(self.origin.name + " to " + self.destination.name)


if __name__ == "__main__":
    # testing info:
    mygraph = Graph("mygraph")
    HI = Node("HI")
    CA = Node("CA")

    mygraph.add_node(HI.name)
    mygraph.add_node(CA.name)

    HI.new_connection(HI, CA, 0.2, 0)

    print(f"Hawaii's connections: ")
    for connection in HI.connections:
        print(connection.return_name(), sep=", ", end="")
    print("\n")

    print(f"Nodes in graph: {mygraph.nodes}")
