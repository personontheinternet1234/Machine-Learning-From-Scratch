import numpy as np
import ast  # Used just for reading preset weights and biases into a list if we want to.

"""
This program uses the nodes structure to practice basic backpropagation.
Made from scratch (No tutorials, no pytorch).
Version: 1.0
Author: Isaac Park Verbrugge, Christian Host-Madsen
"""

# learning presets
learn = "A"  # add this functionality, add ability to choose original weights and biases
epochs = 100000
return_rate = 1000
learning_rate = 0.01
activations = []

# neural network structure
layer_sizes = [2, 3, 2]
layers = len(layer_sizes)
weights = []
biases = []

# training data set
input_training = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
output_training = [
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1]
]


# sigmoid activation function
def sigmoid(values):
    output = 1 / (1 + np.exp(-1 * values))
    return output


# derivative of sigmoid
def sigmoid_prime(values):
    # output = 1 / (1 + np.exp(-1 * values)) * (1 - 1 / (1 + np.exp(-1 * values)))
    output = sigmoid(values) * (1 - sigmoid(values))
    return output


# leaky rectified linear activation function
def relu(values):
    output = np.maximum(0.1 * values, values)
    return output


# derivative of leaky relu
def relu_prime(values):
    return np.where(values > 0, 1, 0.1)


def softmax(values):
    return np.exp(values) / np.sum(np.exp(values))


def cross_entropy(softmax_probs, true_labels):
    true_label_index = np.where(true_labels > 0)[0][0]
    return -np.log(softmax_probs[true_label_index])


def derivative_cross_entropy(values, true_labels):  # derivative is just softmax, unless you are the winner, then it is softmax - 1
    true_label_index = np.where(true_labels > 0)[0][0]

    softmax_probs = softmax(values)
    d_loss_d_values = softmax_probs.copy()
    d_loss_d_values[true_label_index] -= 1
    return d_loss_d_values


# function to reformat data into inputs / correct outputs
def reformat(training_choice):
    inputs = np.reshape(np.array(input_training[training_choice]), (len(input_training[training_choice]), 1))
    expected_values = np.reshape(np.array(output_training[training_choice]), (len(output_training[training_choice]), 1))
    return inputs, expected_values


def xavier_initialize(length, width):
    matrix = np.random.randn(length, width) * np.sqrt(2 / length)
    return matrix


def zeros_initialize(length, width):
    matrix = np.zeros((length, width))
    return matrix


def ones_initialize(length, width):
    matrix = np.ones((length, width))
    return matrix


# forward pass
def forward(inputs):
    global activations
    activations = [inputs]
    for layer in range(layers - 1):
        activation = relu(np.matmul(weights[layer], activations[-1]) + biases[layer])
        activations.append(activation)


# backpropagation
def backward():
    # global expected_values
    # global activations
    # global weights
    # global biases

    d_activations = []
    d_weights = []
    d_biases = []

    # error with respect to last layer
    d_activations.insert(0, -2 * np.subtract(expected_values, activations[-1]))

    for layer in range(layers - 2, -1, -1):  # start at last hidden layer, go back until layer = 0
        # gradient of biases
        d_b = relu_prime(np.matmul(weights[layer], activations[layer]) + biases[layer]) * d_activations[0]
        d_biases.insert(0, d_b)

        # gradient of weights
        upstream = np.resize(d_biases[0], (len(activations[layer]), len(activations[layer + 1]))).T
        local = np.resize(activations[layer].T, (len(activations[layer + 1]), len(activations[layer])))

        d_w = np.multiply(upstream, local)
        d_weights.insert(0, d_w)

        # gradient of activations
        upstream = np.resize(d_biases[0], (len(activations[layer]), len(activations[layer + 1])))
        totals = np.sum(np.multiply(upstream, weights[layer].T), axis=1)

        d_a = np.reshape(totals, (len(activations[layer]), 1))
        d_activations.insert(0, d_a)

    for layer in range(layers - 2, -1, -1):
        weights[layer] = np.subtract(weights[layer], learning_rate * d_weights[layer])
        biases[layer] = np.subtract(biases[layer], learning_rate * d_biases[layer])


# user indexes
input_index = ["a(0)0", "a(0)1"]
output_index = ["checkered", "non checkered"]

while learn != "y" and learn != "n":
    learn = input("Learn? (Y/n): ").lower()

if learn == "y":
    # instantiate weights and biases
    for i in range(layers - 1):
        weights.append(xavier_initialize(layer_sizes[i + 1], layer_sizes[i]))  # Xavier Initialization
        biases.append(zeros_initialize(layer_sizes[i + 1], 1))

    # training loop
    for epoch in range(epochs):
        # choose from training set
        training_choice = int(np.random.rand() * len(input_training))  # SGD choice using np

        # reformat inputs and outputs
        inputs, expected_values = reformat(training_choice)

        # forward pass
        forward(inputs)

        # calculate gradients
        backward()

        # error report
        if epoch % return_rate == 0:
            error = 0
            for i in range(len(input_training)):
                # reformat inputs and outputs
                inputs, expected_values = reformat(i)

                forward(inputs)

                error += np.sum(np.subtract(expected_values, activations[-1]) ** 2)
            print(f"({round((epoch / epochs) * 100)}%) MSE: {error / len(input_training)}")
else:
    with open("etc/weights.txt", "r") as file:
        weights = ast.literal_eval(file.read())
    with open("etc/biases.txt", "r") as file:
        biases = ast.literal_eval(file.read())

    for i in range(len(weights)):
        weights[i] = np.array(weights[i])
    for i in range(len(biases)):
        biases[i] = np.array(biases[i])
print()

save_question = "A"
while save_question != "y" and save_question != "n":
    save_question = input("Save the weights & biases just calculated? (Y/n): ").lower()

if save_question == "y":
    saved_weights = []
    saved_biases = []
    for i in range(len(weights)):
        saved_weights.append(weights[i].tolist())
    for i in range(len(biases)):
        saved_biases.append(biases[i].tolist())

    with open("etc/weights.txt", "w") as file:
        file.write(str(saved_weights))
    with open("etc/biases.txt", "w") as file:
        file.write(str(saved_biases))
else:
    pass

# finalized network application
while True:
    # get inputs
    inputs = []
    for input_node in range(layer_sizes[0]):
        inputs.append(float(input(f"{input_index[input_node]}: ")))

    # forward pass
    inputs = np.reshape(inputs, (len(inputs), 1))
    forward(inputs)

    # result
    print(activations[-1])
    print(f"Outputted: {output_index[np.nanargmax(activations[-1])]}")
