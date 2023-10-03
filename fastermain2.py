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
    return np.where(values > 0, 1, 0.1)


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
epochs = 10000
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


def forward(graph, inputs):
    inputs = np.reshape(np.array(inputs), (len(inputs), 1))

    graph.layers_activations.append(inputs)
    result = np.matmul(graph.layers_weights[0], graph.layers_activations[0])
    result = np.add(result, graph.layers_bias[0])
    result = relu(result)
    graph.layers_activations.append(result)

    for layer in range(1, graph.hidden_layers + 1):  # now repeat for every layer from first hidden going forward
        result = np.matmul(graph.layers_weights[layer], graph.layers_activations[layer])
        result = np.add(result, graph.layers_bias[layer])
        result = relu(result)
        graph.layers_activations.append(result)

    return graph.layers_activations[-1]




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
    d_bias_list.insert(0, relu_prime(np.matmul(weights_list[len(weights_list) - 1], activation_list[len(activation_list) - 2]) + bias_list[len(bias_list) - 1]) *
                       d_activation_list[0])

    # gradient of weights
    d_weights_list.insert(0, np.multiply(np.resize(d_bias_list[len(d_bias_list) - 1], (len(activation_list[len(activation_list) - 2]),
                        len(activation_list[len(activation_list) - 1]))).T, np.resize(activation_list[len(activation_list) - 2].T, (len(activation_list[len(activation_list) - 1]), len(activation_list[len(activation_list) - 2])))))
    # gradient of activations before last layer
    d_activation_list.insert(0, np.reshape(np.sum(np.multiply(np.resize(d_bias_list[0], (len(activation_list[len(activation_list) - 2]),
                            len(activation_list[len(activation_list) - 1]))), weights_list[len(weights_list) - 1].T), axis=1), (len(activation_list[len(activation_list) - 2]), 1)))

    for layer in range(graph.hidden_layers, 0, -1):  # start at last hidden layer, go back until layer = 0
        # gradient of biases
        d_bias_list.insert(0, relu_prime(
            np.matmul(weights_list[layer - 1], activation_list[layer - 1]) + bias_list[layer - 1]) *
                    d_activation_list[0])

        # gradient of weights
        d_weights_list.insert(0, np.multiply(np.resize(d_bias_list[0], (len(activation_list[layer - 1]),
                        len(activation_list[layer]))).T, np.resize(activation_list[layer - 1].T, (len(activation_list[layer]), len(activation_list[layer - 1])))))

        # gradient of activation from before last layer
        d_activation_list.insert(0, np.reshape(np.sum(np.multiply(np.resize(d_bias_list[0], (len(activation_list[layer - 1]),
                                len(activation_list[layer]))), weights_list[layer - 1].T), axis=1), (len(activation_list[layer - 1]), 1)))

    for weight_index in range(len(mygraph.layers_weights)):
        mygraph.layers_weights[weight_index] = np.subtract(mygraph.layers_weights[weight_index], learning_rate * d_weights_list[weight_index])
    for bias_index in range(len(mygraph.layers_bias)):
        mygraph.layers_bias[bias_index] = np.subtract(mygraph.layers_bias[bias_index], learning_rate * d_bias_list[bias_index])

    graph.layers_activations = []


mygraph = Graph()
create_graph(mygraph, 2, 5, 5, 2)

for i in range(epochs):
    # choose from training set
    training_index_choice = random.randint(0, len(input_training) - 1)
    input_choice = input_training[training_index_choice]
    output_choice = output_training[training_index_choice]

    forward(mygraph, input_choice)
    backward(mygraph, output_choice)

correct_outputs = np.reshape(np.array(output_choice), (len(output_choice), 1))
print(f"Error: {np.sum(np.subtract(output_choice, forward(mygraph, input_choice)) ** 2)}")
mygraph.layers_activations = []  # clear activations

# return optimized weights and biases
print(mygraph.layers_weights)
print(mygraph.layers_bias)

# allow user to test optimized network
while True:
    user_test = []
    for i in range(len(mygraph.layers_weights[0][0])):
        activation = float(input(f"activation for node {i}: "))
        user_test.append(activation)
    print(forward(mygraph, user_test))
    mygraph.layers_activations = []  # clear activations
