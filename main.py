import math_functions
import nodes
import random

# Variables in camelCase
# Classes Uppercase Letters
# functions_underscored

actualDataSet = [
    [0,0],
    [0.5,1],
    [1,0]
]
# graph
mygraph = nodes.Graph("mygraph")

# Nodes
input_node = nodes.Node("input")

top_relu = nodes.Node("top_relu", bias=0)

bottom_relu = nodes.Node("bottom_relu", bias=0)

output_node = nodes.Node("output", bias=0)

# connections
input_node.new_connection(input_node, top_relu, weight=round(random.uniform(-5,5), 2))
input_node.new_connection(input_node, bottom_relu, weight=round(random.uniform(-5,5), 2))

top_relu.new_connection(top_relu, output_node, weight=round(random.uniform(-5,5), 2))

bottom_relu.new_connection(bottom_relu, output_node, weight=round(random.uniform(-5,5), 2))

mygraph.add_node(input_node)
mygraph.add_node(top_relu)
mygraph.add_node(bottom_relu)
mygraph.add_node(output_node)

for node in mygraph.nodes:
    for connection in node.connections:
        print(connection.return_name() + " weight: " + str(connection.weight), end=" ")
    print("")

print(f"Nodes in graph: ")
for node in mygraph.nodes:
    print(node.name + " Energy: " + str(node.activationEnergy) + " Bias: " + str(node.bias))