import nodes
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

"""
This program uses the nodes structure to practice basic backpropagation, adjusting a final bias to fit a curve to a set of data points.
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

def softmax(outputEnergies):
    denom = 0
    for i in outputEnergies:
        denom += math.e ** i

    softOutputEnergies = []
    for i in outputEnergies:
        softOutputEnergies.append((math.e**i)/denom)

    return softOutputEnergies

def forward(input):
    input_node.activationEnergy = input

    top_relu.activationEnergy = softplus(input_node.activationEnergy * input_node.connections[0].weight + top_relu.bias)
    bottom_relu.activationEnergy = softplus(input_node.activationEnergy * input_node.connections[1].weight + bottom_relu.bias)

    output_node.activationEnergy = (top_relu.activationEnergy * top_relu.connections[1].weight) + (bottom_relu.activationEnergy * bottom_relu.connections[1].weight) + output_node.bias

    return round(output_node.activationEnergy, 4)

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
def train(i):
    global what_epoch

    if what_epoch < epochs:
        output_node.bias = new_value(actualDataSet, output_node.bias)
        print("new bias: " + str(output_node.bias))

        # plotting machine's data set x and y
        predictedx = []
        predictedy = []
        for i in range(100):
            predictedx.append(i / 100)
            predictedy.append(forward(i / 100))
        plt.cla()
        plt.plot(predictedx, predictedy, label="Predicted pts")

        # plotting actual data set x and y
        plt.plot(convert(actualDataSet)[0], convert(actualDataSet)[1], marker="o", label="Actual Data pts")

        plt.xlim(0,1)
        plt.ylim(-1.5,1)

        what_epoch += 1

what_epoch = 0
ani = FuncAnimation(plt.gcf(), train, interval=100)

plt.legend()
plt.show()

print(softmax([1.43,-0.4,0.23]))
