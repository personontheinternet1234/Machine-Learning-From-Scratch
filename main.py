import networkx as nx
import matplotlib.pyplot as plt

import nodes
import numpy as np
import math
from matplotlib import pyplot as plt

"""
This program uses the nodes structure to practice basic backpropagation.
Made from scratch (No tutorials, no pytorch).
(NetworkX is NOT used for node structure, only to show the neural net as a visual element)
Version: 0.1
Author: Isaac Park Verbrugge
"""


actualDataSet = [
    [0,0],
    [0.5,1],
    [1,0]
]


def softplus(x):  # SoftPlus activation function (NOT USING IT, DIDN'T BUILD IT FOR MATRICES)
    return math.log(1 + math.e**x)


def softmax(outputNodes):
    denom = 0
    for i in outputNodes:
        denom += math.e ** outputNodes[i].activationEnergy

    softOutputEnergies = []
    for i in outputNodes:
        softOutputEnergies.append((math.e**outputNodes[i].activationEnergy)/denom)

    return softOutputEnergies


def sigmoid(x):  # Sigmoid function, built for use with matrices
    y = np.multiply(x, -1)
    y = np.exp(y)
    y = np.add(y, 1)
    y = np.power(y, -1)

    return y


def softmax_derivative():
    # TODO
    ...


def SSR(actualDataSet):  # sum of squared values function
    sum = 0
    for point in actualDataSet:
        ...
        #sum += (forward(point[0]) - point[1]) ** 2 # for each x value
    return sum


G = nx.Graph()


def create_graph(graph, number_input_nodes, number_hidden_layers, number_nodes_per_layer, number_output_nodes):  # graph creation
    name = 0

    # layer matrix sizing
    for iterator in range(number_hidden_layers):
        graph.layers.append([])
        graph.layers_activations.append([])
        graph.layers_bias.append([])

    # input node creation
    for i in range(number_input_nodes):
        graph.layers[0].append(nodes.Node(name, graph, 0, energy=0, bias=0))
        graph.layers_activations[0].append(graph.layers[0][i].activationEnergy)
        graph.layers_bias[0].append(graph.layers[0][i].bias)

        G.add_node(name, pos=(0, i))
        name += 1

    # hidden node creation
    for l in range(number_hidden_layers):
        current_layer = l + 1
        for n in range(number_nodes_per_layer):
            graph.layers[current_layer].append(nodes.Node(name, graph, current_layer, energy=0, bias=0))
            graph.layers_activations[current_layer].append(graph.layers[current_layer][n].activationEnergy)
            graph.layers_bias[current_layer].append(graph.layers[current_layer][n].bias)

            G.add_node(name, pos=(current_layer, n))
            name += 1

    # output node creation
    for o in range(number_output_nodes):
        last_layer = len(graph.layers) - 1
        graph.layers[last_layer].append(nodes.Node(name, graph, last_layer, energy=0, bias=0))
        graph.layers_activations[last_layer].append(graph.layers[last_layer][o].activationEnergy)
        graph.layers_bias[last_layer].append(graph.layers[last_layer][o].bias)

        G.add_node(name, pos=(last_layer, o))
        name += 1

    # connections
    for i in range(len(graph.layers) - 1):
        for originnode in graph.layers[i]:
            for destinationnode in graph.layers[i + 1]:
                originnode.new_connection(originnode, destinationnode, np.random.normal(0, 5))

                G.add_edge(int(originnode.name),int(destinationnode.name))

    # np stuff for np.matmult()
    for i in range(len(graph.layers)):
        graph.layers[i] = np.array(graph.layers[i])
        graph.layers_activations[i] = np.array(graph.layers_activations[i], dtype=object)
        graph.layers_bias[i] = np.array(graph.layers_bias[i])


def forward(graph, inputs):  # forward pass

    def inputstep(inputs):
        # flips inputs so instead of being 1 x n it will be n x 1
        inputs = np.array(inputs).reshape((len(inputs), 1))

        # sets activationEnergies of each object in inputs if we need them for backprop
        for node in range(len(graph.layers[0])):
            graph.layers[0][node].activationEnergy = inputs[node][0]

        # sets the matrix in graphs representing inputs to the n x 1 inputs we found earlier
        graph.layers_activations[0] = inputs

        # input layer to next layer calc
        layerweights_initial = []
        for node in graph.layers[0]:
            layerweights_initial += [node.connections_weights]
        layerweights_initial = flipmatrix(layerweights_initial)

        # weight step
        result = np.matmul(layerweights_initial, graph.layers_activations[0])

        # bias step
        result = result + graph.layers_bias[1].reshape((len(graph.layers_bias[1]), 1))

        # sigmoid step
        result = sigmoid(result)

        return result

    def layerstep(result):
        for l in range(len(graph.layers) - 2):  # -2: 1 disregarded bc of input, 1 disregarded bc output
            current_layer = l + 1

            # sets activationEnergies of each object in each layer if we need them for backprop
            for node in range(len(graph.layers[current_layer])):
                graph.layers[current_layer][node].activationEnergy = result[node][0]

            # sets the matrix in graphs representing energies of the current layer to the n x 1 results of the last calc
            graph.layers_activations[current_layer] = result

            # current layer to next layer calc
            layerweights = []
            for node in graph.layers[current_layer]:  # nodes starting after [0] (inputs)
                layerweights += [node.connections_weights]
            layerweights = flipmatrix(layerweights)

            # weight step
            result = np.matmul(layerweights, graph.layers_activations[current_layer])

            # bias step
            result = result + graph.layers_bias[current_layer + 1].reshape((len(graph.layers_bias[current_layer + 1]), 1))

            # sigmoid step
            result = sigmoid(result)

            return result

    def last_step(result):
        last_layer = len(graph.layers) - 1
        # sets activationEnergies of each object in each layer if we need them for backprop
        for node in range(len(graph.layers[last_layer])):
            graph.layers[last_layer][node].activationEnergy = result[node][0]

        # sets the matrix in graphs representing energies of the current layer to the n x 1 results of the last calcs
        graph.layers_activations[last_layer] = result

        return result

    # this is kinda crazy. Put 'em all into functions so it'd be nicer to read. Hope it still works!
    return last_step(layerstep(inputstep(inputs)))


def backward():
    # usage: node1-2 = add_backprop(node1-2.upstreamValue())
    def multiply_backprop(upstream_gradient, downstream_nodes, target_node):
        product = upstream_gradient
        for node in downstream_nodes:
            product *= node.activationEnergy
        product /= target_node.activationEnergy

        return product
    def max_backprop(upstream_gradient, downstream_nodes, target_node):
        largest = downstream_nodes[0].activationEnergy
        for node in downstream_nodes:
            if node.activationEnergy >= largest:
                largest = node.activationEnergy

        if target_node.activationEnergy == largest:
            return upstream_gradient
        else:
            return 0
    def copy_backprop(upstream_gradients):  # upstream_gradients is a list tho... Idk where imma use copy_backprop
        sum = 0
        for i in upstream_gradients:
            sum += i
        return sum
    def add_backprop(upstream_gradient):
        return(upstream_gradient)


def new_value(actualDataSet, oldWeight):  # gradient descent function for a given connection's weight.
    # take derivitive of sum of squared values with respect to w4:
    # sum of each data point: 2 * (observed - predicted) * -1

    sum = 0
    for point in actualDataSet:
        ...
        #sum += -2 * (point[1] - forward(point[0]))
    print("SSR (should approach 0): " + str(SSR(actualDataSet)))

    # returns next weight or bias the connection should have
    #return round(oldWeight - (sum * learningRate), 4)


# Just to convert my weird usage of actualDataSet being a matrix.
def convert(dataset):
    data = np.array(dataset)
    return [data[:, 0], data[:, 1]]


# Function for flipping dimensions of a matrix (This was unironically quite tough to make because my laptop died and I
# had to do it in my head.
def flipmatrix(in_matrix):
    out_matrix = []

    for i in in_matrix[0]:
        out_matrix.append([i])
    for i in range(len(in_matrix) - 1):
        for x in range(len(in_matrix[i + 1])):
            out_matrix[x].append(in_matrix[i + 1][x])
    return np.array(out_matrix)


mygraph = nodes.Graph("mygraph")
# graph, number_input_nodes, number_hidden_layers, number_nodes_per_layer, number_output_nodes
create_graph(mygraph, 2, 1, 2, 5)

# print(mygraph.layers)
# for layer in range(len(mygraph.layers)):
#     for node in mygraph.layers[layer]:
#         for connection in node.connections:
#             print("layer" + str(layer) + " " + str(connection.origin.name) + " to layer" +
#                   str(connection.destination.layer) + " " + str(connection.destination.name))
# print()
# for layer in range(len(mygraph.layers)):
#     for node in mygraph.layers[layer]:
#         for back_connection in node.back_connections:
#             print("layer" + str(layer) + " " + str(back_connection.origin.name) + " to layer" +
#                   str(back_connection.destination.layer) + " " + str(back_connection.destination.name))

print(forward(mygraph, [0, 1]))

pos=nx.get_node_attributes(G,'pos')
nx.draw(G, pos, with_labels=True)
plt.show()