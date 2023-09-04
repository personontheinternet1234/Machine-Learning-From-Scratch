import nodes
import numpy as np
import math
from matplotlib import pyplot as plt

"""
This program uses the nodes structure to practice basic backpropagation.
Made from scratch (No tutorials, no pytorch).
Version: 0.1
Author: Isaac Park Verbrugge
"""

actualDataSet = [
    [0,0],
    [0.5,1],
    [1,0]
]


def softplus(x):  # SoftPlus activation function
    return math.log(1 + math.e**x)

def softmax(outputNodes):
    denom = 0
    for i in outputNodes:
        denom += math.e ** outputNodes[i].activationEnergy

    softOutputEnergies = []
    for i in outputNodes:
        softOutputEnergies.append((math.e**outputNodes[i].activationEnergy)/denom)

    return softOutputEnergies

def softmax_derivative():
    # TODO
    ...

def create_graph(graph_name, number_input_nodes, number_hidden_layers, number_nodes_per_layer, number_output_nodes):
    graph_name = nodes.Graph(graph_name)

    #node creation
    for i in range(len(number_input_nodes)):
        graph_name.nodes.append(nodes.Node(i, graph_name, 0, energy=0, bias=0))

    for l in range(len(number_hidden_layers)):
        for n in range(len(number_nodes_per_layer)):
            graph_name.nodes.append(nodes.Node(n, graph_name, l, energy=0, bias=0))

    for o in range(len(number_output_nodes)):
        graph_name.nodes.append(nodes.Node(o, graph_name, l + 1, energy=0, bias=0))

    #connections
    for i in range(len(number_input_nodes)):
        graph_name.nodes.append(nodes.Node(i, graph_name, 0, energy=0, bias=0))

    for l in range(len(number_hidden_layers)):
        for n in range(len(number_nodes_per_layer)):
            graph_name.nodes.append(nodes.Node(n, graph_name, l, energy=0, bias=0))

    for o in range(len(number_output_nodes)):
        graph_name.nodes.append(nodes.Node(o, graph_name, l + 1, energy=0, bias=0))


def forward(input):
    # input = [1,2,3...784]
    top_relu.activationEnergy = softplus(input_node.activationEnergy * input_node.connections[0].weight + top_relu.bias)
    bottom_relu.activationEnergy = softplus(input_node.activationEnergy * input_node.connections[1].weight + bottom_relu.bias)

    output_node.activationEnergy = (top_relu.activationEnergy * top_relu.connections[0].weight) + (bottom_relu.activationEnergy * bottom_relu.connections[0].weight) + output_node.bias

    return round(output_node.activationEnergy, 4)

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


def SSR(actualDataSet):  # sum of squared values function
    sum = 0
    for point in actualDataSet:
        sum += (forward(point[0]) - point[1]) ** 2 # for each x value
    return sum


def new_value(actualDataSet, oldWeight):  # gradient descent function for a given connection's weight.
    # take derivitive of sum of squared values with respect to w4:
    # sum of each data point: 2 * (observed - predicted) * -1

    sum = 0
    for point in actualDataSet:
        sum += -2 * (point[1] - forward(point[0]))
    print("SSR (should approach 0): " + str(SSR(actualDataSet)))

    # returns next weight or bias the connection should have
    return round(oldWeight - (sum * learningRate), 4)

# Just to convert my weird usage of actualDataSet being a matrix.
def convert(dataset):
    data = np.array(dataset)
    return [data[:, 0], data[:, 1]]

#########################################

# graph
mygraph = nodes.Graph("mygraph")

# Nodes
input_node = nodes.Node("input", mygraph, 0)

top_relu = nodes.Node("top_relu", mygraph, 1, bias=-1.43)

bottom_relu = nodes.Node("bottom_relu", mygraph, 1, bias=0.57)

output_node = nodes.Node("output", mygraph, 2, bias=0)

# connections
input_node.new_connection(input_node, top_relu, weight=3.34)
input_node.new_connection(input_node, bottom_relu, weight=-3.53)

top_relu.new_connection(top_relu, output_node, weight=-1.22)

bottom_relu.new_connection(bottom_relu, output_node, weight=-2.30)

#adding to graph
mygraph.add_node(input_node)
mygraph.add_node(top_relu)
mygraph.add_node(bottom_relu)
mygraph.add_node(output_node)

#########################################

# prompts user for iterations and learning rate
epochs = 0
learningRate = 0
while epochs <= 0:
    epochs = int(input("How many epochs: "))
while learningRate <= 0 or learningRate >= 1:
    learningRate = float(input("Learning Rate: "))

# training last bias.
for i in range(epochs):
    output_node.bias = new_value(actualDataSet, output_node.bias)
    print("new bias: " + str(output_node.bias))

# plotting actual data set x and y
plt.plot(convert(actualDataSet)[0], convert(actualDataSet)[1])

# plotting machine's data set x and y
predictedx = []
predictedy = []
for i in range(100):
    predictedx.append(i/100)
    predictedy.append(forward(i/100))
plt.plot(predictedx, predictedy)

plt.show()

print(softmax([1.43,-0.4,0.23]))
