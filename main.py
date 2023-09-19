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

# [inputnodevalues,outputnodeindexwewant]
actualDataSet = [
    [[1,0],0],
    [[0,1],1],
]


# Just to convert my weird usage of actualDataSet being a matrix.
def convert(dataset):
    data = np.array(dataset)
    return [data[:, 0], data[:, 1]]


def softmax(outputnodesactivations):
    denom = 0
    eoutputnodesactivations = np.exp(outputnodesactivations)

    for i in eoutputnodesactivations:
        denom += i[0]

    smaxoutputs = np.divide(eoutputnodesactivations, denom)
    return smaxoutputs


def crossentropy(smaxoutputs, correct_output):
    centropyoutput = -1 * math.log(smaxoutputs[correct_output][0])
    return centropyoutput
    # Usage: crossentropy( softmax(forward(mygraph, inputs) )[ActuallyCorrectNodeIndex][0])


def softplus(x):
    y = np.exp(x)
    y = np.add(x, 1)
    y = np.log(x)
    return y

def sigmoid(x):  # Sigmoid function, built for use with matrices
    y = np.multiply(x, -1)
    y = np.exp(y)
    y = np.add(y, 1)
    y = np.power(y, -1)

    return y


def relu(x):  # Relu function, built for use with matrices
    return x * (x > 0)


def derivativelossrespecttosomething(correct_guy, centropyoutput, outputs):
    # usage: derivativelossrespecttosomething(correct index, crossentropy(softmax(output node energies), correct index), output node energies)
    results = []

    denom = 0
    for output in outputs:
        denom -= math.exp(output[0])

    for output in outputs:
        if output[0] == outputs[correct_guy][0]:  # if the output is the one that it should be
            num = 0
            for o in outputs:
                num += math.exp(o[0])
            num -= math.exp(output[0])
            results.append(num / denom)
        else:
            results.append(math.exp(output[0]) / denom)
    return results


G = nx.Graph()


def create_graph(graph, number_input_nodes, number_hidden_layers, number_nodes_per_layer, number_output_nodes):  # graph creation
    name = 0

    # layer matrix sizing
    for iterator in range(number_hidden_layers):
        graph.layers.append([])
        graph.layers_activations.append([])

    # input node creation
    for i in range(number_input_nodes):
        graph.layers[0].append(nodes.Node(name, graph, 0, energy=0, bias=0))
        graph.layers_activations[0].append(graph.layers[0][i].activationEnergy)

        G.add_node(name, pos=(0, i))
        name += 1

    # hidden node creation
    for l in range(number_hidden_layers):
        current_layer = l + 1
        for n in range(number_nodes_per_layer):
            graph.layers[current_layer].append(nodes.Node(name, graph, current_layer, energy=0, bias=0))
            graph.layers_activations[current_layer].append(graph.layers[current_layer][n].activationEnergy)

            G.add_node(name, pos=(current_layer, n))
            name += 1

    # output node creation
    for o in range(number_output_nodes):
        last_layer = len(graph.layers) - 1
        graph.layers[last_layer].append(nodes.Node(name, graph, last_layer, energy=0, bias=0))
        graph.layers_activations[last_layer].append(graph.layers[last_layer][o].activationEnergy)

        G.add_node(name, pos=(last_layer, o))
        name += 1

    # connections
    for i in range(len(graph.layers) - 1):
        for originnode in graph.layers[i]:
            for destinationnode in graph.layers[i + 1]:
                originnode.new_connection(originnode, destinationnode, np.random.normal(0, 5))

                G.add_edge(int(originnode.name),int(destinationnode.name))
            originnode.fix_connections_weights()

    # np stuff for np.matmult()
    for i in range(len(graph.layers)):
        graph.layers[i] = np.array(graph.layers[i])
        graph.layers_activations[i] = np.array(graph.layers_activations[i], dtype=object)


def forward(graph, inputs):  # forward pass

    def inputstep(inputs):
        # flips inputs so instead of being 1 x n it will be n x 1
        inputs = np.array(inputs).reshape((len(inputs), 1))

        # sets activationEnergies of each object in inputs if we need them for backprop
        for node in range(len(graph.layers[0])):
            graph.layers[0][node].activationEnergy = inputs[node][0]

        # sets the matrix in graphs representing inputs to the n x 1 inputs we found earlier
        graph.layers_activations[0] = inputs

        # makes a np matrix of all the weights we're going to need
        layerweights_initial = []
        for node in graph.layers[0]:
            layerweights_initial += [node.connections_weights]
        layerweights_initial = flipmatrix(layerweights_initial)

        # makes a np matrix of all the biases we're going to need
        bias_initialplusone = []
        for node in graph.layers[1]:
            bias_initialplusone.append([node.bias])
        bias_initialplusone = np.array(bias_initialplusone)

        # weight step
        result = np.matmul(layerweights_initial, graph.layers_activations[0])

        # bias step
        result = result + bias_initialplusone

        # softplus step
        result = softplus(result)

        return result

    def layerstep(result):
        for l in range(len(graph.layers) - 2):  # -2: 1 disregarded bc of input, 1 disregarded bc output
            current_layer = l + 1

            # sets activationEnergies of each object in each layer if we need them for backprop
            for node in range(len(graph.layers[current_layer])):
                graph.layers[current_layer][node].activationEnergy = result[node][0]

            # sets the matrix in graphs representing energies of the current layer to the n x 1 results of the last calc
            graph.layers_activations[current_layer] = result

            # sets nodes energies to result
            for node in range(len(graph.layers[current_layer])):
                graph.layers[current_layer][node].energy = result[node][0]

            # makes a np matrix of all the weights we're going to need
            layerweights = []
            for node in graph.layers[current_layer]:  # nodes starting after [0] (inputs)
                layerweights += [node.connections_weights]
            layerweights = flipmatrix(layerweights)

            # makes a np matrix of all the biases we're going to need
            bias_plusone = []
            for node in graph.layers[current_layer + 1]:
                bias_plusone.append([node.bias])
            bias_plusone = np.array(bias_plusone)

            # weight step
            result = np.matmul(layerweights, graph.layers_activations[current_layer])

            # bias step
            result = result + bias_plusone

            # softplus step
            result = softplus(result)

        return result

    def last_step(result):
        last_layer = len(graph.layers) - 1
        # sets activationEnergies of each object in each layer if we need them for backprop

        for node in range(len(graph.layers[last_layer])):
            graph.layers[last_layer][node].activationEnergy = result[node][0]

        # sets the matrix in graphs representing energies of the current layer to the n x 1 results of the last calcs
        graph.layers_activations[last_layer] = result

        # sets nodes energies to the result
        for node in range(len(graph.layers[last_layer])):
            graph.layers[last_layer][node].energy = result[node][0]

        return result

    # this is kinda crazy. Put 'em all into functions, so it'd be nicer to read. Hope it still works!
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


def weightupdate(connection_in_question, new_weight):
    connection_in_question.weight = new_weight
    connection_in_question.origin.fix_connections_weights()


def biasupdate(node_in_question, new_bias):
    node_in_question.bias = new_bias


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

def testgraphcreate():
    # graph, number_input_nodes, number_hidden_layers, number_nodes_per_layer, number_output_nodes
    create_graph(mygraph, 2, 1, 3, 2)

    # Node 1
    weightupdate(mygraph.layers[0][0].connections[0], 1)
    weightupdate(mygraph.layers[0][0].connections[1], 1)
    weightupdate(mygraph.layers[0][0].connections[2], 1)

    # Node 2
    weightupdate(mygraph.layers[0][1].connections[0], 1)
    weightupdate(mygraph.layers[0][1].connections[1], 1)
    weightupdate(mygraph.layers[0][1].connections[2], 1)

    # Node 3
    biasupdate(mygraph.layers[1][0], 0)
    weightupdate(mygraph.layers[1][0].connections[0], 1)
    weightupdate(mygraph.layers[1][0].connections[1], 1)

    # Node 4
    biasupdate(mygraph.layers[1][1], 0)
    weightupdate(mygraph.layers[1][1].connections[0], 1)
    weightupdate(mygraph.layers[1][1].connections[1], 1)

    # Node 5
    biasupdate(mygraph.layers[1][2], 0)
    weightupdate(mygraph.layers[1][2].connections[0], 1)
    weightupdate(mygraph.layers[1][2].connections[1], 1)

    # Node 6
    biasupdate(mygraph.layers[2][0], 0)

    # Node 7
    biasupdate(mygraph.layers[2][1], 0)


testgraphcreate()
calculatedoutputs = forward(mygraph, [1,1])
print(calculatedoutputs)
print(derivativelossrespecttosomething(0, crossentropy(softmax(calculatedoutputs), 0), calculatedoutputs))

pos=nx.get_node_attributes(G,'pos')
nx.draw(G, pos, with_labels=True)
plt.show()


