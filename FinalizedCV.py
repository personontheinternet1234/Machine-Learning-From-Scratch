import numpy as np
import random
import ast
from tensorflow import keras

"""
This program uses the nodes structure to practice basic backpropagation.
Made from scratch (No tutorials, no pytorch).
Version: 1.0
Author: Isaac Park Verbrugge, Christian Host-Madsen
"""
activations = []


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


# function to reformat data into inputs / correct outputs
def reformat(training_choice):
    inputs = np.reshape(np.array(input_training[training_choice]), (len(input_training[training_choice]), 1))
    expected_values = np.reshape(np.array(output_training[training_choice]), (len(output_training[training_choice]), 1))
    return inputs, expected_values


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
    d_a_last = -2 * np.subtract(expected_values, activations[-1])
    d_activations.insert(0, d_a_last)

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
        # print(d_weights[layer])
        # print("\n\n\n")
        # print(d_biases[layer])

        weights[layer] = np.subtract(weights[layer], learning_rate * d_weights[layer])
        biases[layer] = np.subtract(biases[layer], learning_rate * d_biases[layer])


# user indexes
input_index = ["a(0)0", "a(0)1"]
output_index = ["1", "2"]

# learning presets
learn = True  # add this functionality, add ability to choose original weights and biases
non_linearity = "relu"  # add this functionality
error_analysis = "SSR"  # add this functionality
epochs = 100000
return_rate = 1000
learning_rate = 0.00000001

(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()

# training data set
input_training = [

]

output_training = [

]



for i in range(1000):
    input_training.append(train_x[i].flatten().tolist())

    node_values = []
    for h in range(10):
        if h == train_y[i]:
            node_values.append(255)
        else:
            node_values.append(0)
    output_training.append(node_values)

# neural network structure
layer_sizes = [784, 10, 10, 10]
layers = len(layer_sizes)
weights = []
biases = []

while learn != "y" and learn != "n":
    learn = input("Learn? (Y/n): ").lower()

if learn == "y":
    # instantiate weights and biases
    for i in range(layers - 1):
        weights.append(np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * np.sqrt(2 / layer_sizes[i]))  # Xavier Initialization
        biases.append(np.zeros((layer_sizes[i + 1], 1)))

    # training loop
    for epoch in range(epochs):
        # choose from training set
        # training_choice = int(np.random.rand() * len(input_training))  # Random int from 0 - len(input_training) using np
        training_choice = random.randint(0, len(input_training) - 1)

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
                inputs, expected_values = reformat(training_choice)

                forward(inputs)

                error += np.sum(np.subtract(expected_values, activations[-1]) ** 2)
            print(f"({round((epoch / epochs) * 100)}%) MSE: {error / len(input_training)}")
else:
    with open("adjustables/weights.txt", "r") as file:
        weights = ast.literal_eval(file.read())
    with open("adjustables/biases.txt", "r") as file:
        biases = ast.literal_eval(file.read())

    for i in range(len(weights)):
        weights[i] = np.array(weights[i])
    for i in range(len(biases)):
        biases[i] = np.array(biases[i])
print()

save_question = "A"
while save_question != "y" and save_question != "n":
    save_question = input("Save the weights & biases just calculated? (Y/n): ").lower()

    saved_weights = []
    saved_biases = []
    for i in range(len(weights)):
        saved_weights.append(weights[i].tolist())
    for i in range(len(biases)):
        saved_biases.append(biases[i].tolist())

    with open("adjustables/weights.txt", "w") as file:
        file.write(str(saved_weights))
    with open("adjustables/biases.txt", "w") as file:
        file.write(str(saved_biases))

# finalized network application
while True:
    # get inputs
    choice = int(input(f"Image Choice #: "))
    inputs = train_x[choice].flatten().tolist()

    # forward pass
    inputs = np.reshape(inputs, (len(inputs), 1))
    forward(inputs)

    # result
    print(activations[-1])
    print(f"Should be: {train_y[choice]}")
    print(f"Outputted: {np.nanargmax(activations[-1])}")
