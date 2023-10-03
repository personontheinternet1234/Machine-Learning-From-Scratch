import numpy as np
import random
from tqdm import tqdm


# NeuralNet.py
# Isaac Verbrugge & Christian Host-Madsen
# (program info)

# function definitions


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

        # one for inputs, one for outputs
        self.layers_d_activations = [

        ]
        # don't need one for inputs
        self.layers_d_weights = [

        ]
        # one for outputs
        self.layers_d_bias = [

        ]


def create_graph(graph, input_nodes, hidden_layers, hidden_layers_nodes, output_nodes):
    graph.hidden_layers = hidden_layers

    # weights for input
    graph.layers_weights.append(np.random.randn(hidden_layers_nodes, input_nodes) * np.sqrt(2 / input_nodes))  # Xavier initialization

    for layer in range(hidden_layers):
        # Creating list(s) per layer to hold our activation vectors.
        # graph.layers_activations.append(np.array([]))

        # Creating a bias vector for each layer
        graph.layers_bias.append(np.zeros((hidden_layers_nodes, 1)))

        if layer != 0:
            # weights for hidden layers, except last to output cus that is different dimensions
            graph.layers_weights.append(np.random.randn(hidden_layers_nodes, hidden_layers_nodes) * np.sqrt(2 / hidden_layers_nodes))

    # weights for last hidden layer to outputs
    graph.layers_weights.append(np.random.randn(output_nodes, hidden_layers_nodes) * np.sqrt(2 / hidden_layers_nodes))

    # biases for last layer
    graph.layers_bias.append(np.zeros((output_nodes, 1)))


def sigmoid(values):
    output = 1 / (1 + np.exp(-1 * values))
    return output


def sigmoid_prime(values):
    # output = 1 / (1 + np.exp(-1 * values)) * (1 - 1 / (1 + np.exp(-1 * values)))
    output = sigmoid(values) * (1 - sigmoid(values))
    return output


def relu(values):
    output = np.maximum(0.1 * values, values)
    return output


def relu_prime(values):
    if values > 0:
        return 1
    else:
        return 0.1


def forward_pass(a0, w0, b1, w1, b2):
    # forward pass
    a1 = sigmoid(np.matmul(w0, a0) + b1)
    a2 = sigmoid(np.matmul(w1, a1) + b2)

    return a2


# user output index
user_outputs = ["1", "2"]

# neural network structure
input_size = 2
hidden_layers = 2  # update for this later
hidden_sizes = [3, 2]  # update for this later
hidden_1_size = 3  # phase this out
output_size = 2

# learning presets
learn = True  # add this functionality, add ability to choose original weights and biases
epochs = 100000
return_rate = 1000
learning_rate = 0.01

# training data
input_training = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
output_training = [
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
]

# instantiate weights and biases
w0 = np.random.randn(hidden_1_size, input_size)  # going_to x from
b1 = np.zeros((hidden_1_size, 1))
w1 = np.random.randn(output_size, hidden_1_size)
b2 = np.zeros((output_size, 1))


def forward(graph, inputs):
    inputs = np.reshape(np.array(inputs), (len(inputs), 1))

    graph.layers_activations.append(inputs)
    result = np.matmul(graph.layers_weights[0], graph.layers_activations[0])
    result = np.add(result, graph.layers_bias[0])
    result = sigmoid(result)
    graph.layers_activations.append(result)

    for layer in range(1, graph.hidden_layers + 1):  # now repeat for every layer from first hidden going forward
        result = np.matmul(graph.layers_weights[layer], graph.layers_activations[layer])
        result = np.add(result, graph.layers_bias[layer])
        result = sigmoid(result)
        graph.layers_activations.append(result)


def backward():
    ...


mygraph = Graph()
create_graph(mygraph, 2, 3, 3, 4)


epochs = 1
for i in range(epochs):
    # choose from training set
    training_index_choice = random.randint(0, len(input_training) - 1)
    input_choice = input_training[training_index_choice]
    output_choice = output_training[training_index_choice]

    forward(mygraph, input_choice)
    print(mygraph.layers_activations)
    print(len(mygraph.layers_activations[len(mygraph.layers_activations) - 2]))
    backward(mygraph, output_choice)

    # error report every few epochs
    #     if i % return_rate == 0:
    #         error = 0
    #         for j in range(len(a2)):
    #             error += (c[j] - a2[j]) ** 2
    #         print(f"Error: {error} ({round(i / epochs * 100)}%)")


def backward(graph, correct_outputs):
    correct_outputs = np.reshape(np.array(correct_outputs), (len(correct_outputs), 1))
    activation_list = graph.layers_activations
    weights_list = graph.layers_weights
    bias_list = graph.layers_bias

    d_activation_list = graph.layers_d_activations
    d_weights_list = graph.layers_d_weights
    d_bias_list = graph.layers_d_bias

    # error with respect to last layer
    d_activation_list.insert(0, -2 * np.subtract(correct_outputs, activation_list[len(activation_list) - 1]))

    # gradient of biases
    d_bias_list.insert(0, sigmoid_prime(np.matmul(weights_list[len(weights_list) - 1], activation_list[len(activation_list) - 2]) + bias_list[len(bias_list) - 1]) *
                       d_activation_list[len(d_activation_list) - 1])

    # gradient of weights
    d_weights_list.insert(0, np.multiply(np.resize(d_bias_list[len(d_bias_list) - 1], (len(activation_list[len(activation_list) - 2]),
                        len(activation_list[len(activation_list) - 1]))).T, np.resize(activation_list[len(activation_list) - 2].T, (len(activation_list[len(activation_list) - 2]), len(activation_list[len(activation_list) - 1])))))

    for l in graph.hidden_layers

#     # calculate gradients
#
#     # second layer
#     d_a2 = -2 * np.subtract(c, a2)
#     d_b2 = sigmoid_prime(np.matmul(w1, a1) + b2) * d_a2
#     d_w1 = np.multiply(np.resize(d_b2, (hidden_1_size, output_size)).T, np.resize(a1.T, (output_size, hidden_1_size)))
#
#     # first layer
#     d_a1 = np.reshape(np.sum(np.multiply(np.resize(d_b2, (hidden_1_size, output_size)), w1.T), axis=1),
#                       (hidden_1_size, 1))
#     d_b1 = sigmoid_prime(np.matmul(w0, a0) + b1) * d_a1
#     d_w0 = np.multiply(np.resize(d_b1, (input_size, hidden_1_size)).T, np.resize(a0.T, (hidden_1_size, input_size)))
#
#     # optimize weights and biases
#
#     w0 = np.subtract(w0, learning_rate * d_w0)
#     b1 = np.subtract(b1, learning_rate * d_b1)
#     w1 = np.subtract(w1, learning_rate * d_w1)
#     b2 = np.subtract(b2, learning_rate * d_b2)
#
#     # error report every few epochs
#
#     if i % return_rate == 0:
#         error = 0
#         for j in range(len(a2)):
#             error += (c[j] - a2[j]) ** 2
#         print(f"Error: {error} ({round(i / epochs * 100)}%)")
#
# # return optimized weights and biases
# print("")
# print("w0:")
# print(w0)
# print("b1:")
# print(b1)
# print("w1:")
# print(w1)
# print("b2:")
# print(b2)
#
# # allow user to test optimized network
# while True:
#     print("")
#     inputs = []
#     for i in range(input_size):
#         inputs.append(float(input("a(0)" + str(i) + ": ")))  # gets inputs
#
#     # forward pass
#     a0 = np.reshape(inputs, (len(inputs), 1))
#     a2 = forward_pass(a0, w0, b1, w1, b2)
#
#     # result
#     print("")
#     print(a2)
#     output_index = np.nanargmax(np.where(a2 == a2.max(), a2, np.nan))
#     print(f"Predicted: {user_outputs[output_index]}")
