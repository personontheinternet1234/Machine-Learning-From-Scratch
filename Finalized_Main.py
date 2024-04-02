import ast
import random
import time

from matplotlib import pyplot as plt
import numpy as np
# import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

"""
This program uses the nodes structure to practice basic backpropagation.
Made from scratch (No tutorials, no pytorch).
Version: 1.0
Author: Isaac Park Verbrugge, Christian Host-Madsen
"""

# sigmoid activation function
def sigmoid(values):
    output = 1 / (1 + np.exp(-1 * values))
    return output


# derivative of sigmoid
def d_sigmoid(values):
    # output = 1 / (1 + np.exp(-1 * values)) * (1 - 1 / (1 + np.exp(-1 * values)))
    output = sigmoid(values) * (1 - sigmoid(values))
    return output


# leaky rectified linear activation function
def l_relu(values):
    output = np.maximum(0.1 * values, values)
    return output


# derivative of leaky relu
def d_l_relu(values):
    return np.where(values > 0, 1, 0.1)


def softmax(values):
    return np.exp(values) / np.sum(np.exp(values))


# cross entropy
def centropy(softmax_probs, true_labels):
    true_label_index = np.where(true_labels > 0)[0][0]
    return -np.log(softmax_probs[true_label_index])


# derivative of cross entropy
def d_centropy(values, true_labels):
    # derivative is just softmax, unless you are the winner, then it is softmax - 1
    true_label_index = np.where(true_labels > 0)[0][0]
    softmax_probs = softmax(values)
    d_loss_d_values = softmax_probs.copy()
    d_loss_d_values[true_label_index] -= 1
    return d_loss_d_values


def vectorize(list):
    # turn a list into a vector
    vector = np.reshape(np.array(list), (len(list), 1))
    return vector


# xavier initialized array
def xavier_initialize(length, width):
    matrix = np.random.randn(length, width) * np.sqrt(2 / length)
    return matrix


# zeros array
def zeros_initialize(length, width):
    matrix = np.zeros((length, width))
    return matrix


# forward pass
def forward(inputs, weights, biases):
    activations = [inputs]
    for layer in range(layers - 1):
        activation = l_relu(np.matmul(weights[layer], activations[-1]) + biases[layer])
        activations.append(activation)
    return activations, weights, biases


# backpropagation
def backward(activations, weights, biases, true):
    # initialize lists
    d_activations = []
    d_weights = []
    d_biases = []

    # error with respect to last layer
    d_activations.insert(0, -2 * np.subtract(true, activations[-1]))

    # loop through layers backwards
    for layer in range(layers - 2, -1, -1):
        # gradient of biases
        d_b = d_l_relu(np.matmul(weights[layer], activations[layer]) + biases[layer]) * d_activations[0]
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
        # weights[layer] = np.subtract(weights[layer], learning_rate * (d_weights[layer] + (0.1 / 4) * weights[layer]))
        biases[layer] = np.subtract(biases[layer], learning_rate * d_biases[layer])

    return activations, weights, biases


# confusion matrix
def plot_cm(cm, title=None, labels=None, color="binary"):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )
    disp.plot(cmap=color)
    disp.ax_.set_title(title)
    plt.show()


# normal graph
def plot_graph(data, title=None, labels=None, color="black"):
    plt.plot(data[0], data[1], color=color)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)
    plt.show()


# network settings
learn = True
load = False
save = False
generate_graphs = True
epochs = 10000
return_rate = 1
learning_rate = 0.001
lambda_reg = 0.1

# network structure
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

# user indexes
input_index = ["a(0)0", "a(0)1"]
output_index = ["checkered", "non checkered"]

""" network code """

start_time = time.time()

# reformat training data
for i in range(len(input_training)):
    input_training[i] = vectorize(input_training[i])
for i in range(len(output_training)):
    output_training[i] = vectorize(output_training[i])

# load weights and biases
if load:
    with open("etc/weights.txt", "r") as file:
        weights = ast.literal_eval(file.read())
    with open("etc/biases.txt", "r") as file:
        biases = ast.literal_eval(file.read())
    for i in range(len(weights)):
        weights[i] = np.array(weights[i])
    for i in range(len(biases)):
        biases[i] = np.array(biases[i])
else:
    # instantiate weights and biases
    for i in range(layers - 1):
        weights.append(xavier_initialize(layer_sizes[i + 1], layer_sizes[i]))
        biases.append(zeros_initialize(layer_sizes[i + 1], 1))

# training loop
if learn:
    saved_epochs = []
    saved_errors = []
    # instantiate weights and biases
    for i in range(layers - 1):
        weights.append(xavier_initialize(layer_sizes[i + 1], layer_sizes[i]))  # Xavier Initialization
        biases.append(zeros_initialize(layer_sizes[i + 1], 1))

    # training loop
    for epoch in range(epochs):
        # choose from training set
        training_choice = int(np.random.rand() * len(input_training))  # SGD choice using np

        # reformat inputs and outputs
        inputs = input_training[training_choice]
        true = output_training[training_choice]

        # forward pass
        activations, weights, biases = forward(inputs, weights, biases)

        # calculate gradients
        activations, weights, biases = backward(activations, weights, biases, true)

        # error report
        if epoch % return_rate == 0:
            error = 0
            for i in range(len(input_training)):
                activations, _, _ = forward(inputs, weights, biases)
                error += np.sum(np.subtract(true, activations[-1]) ** 2)
            # print(f"({round((epoch / epochs) * 100)}%) MSE: {error / len(input_training)}")
            saved_epochs.append(epoch)
            saved_errors.append(error)

end_time = time.time()

""" return results """

# calculate accuracies
error = 0
correct = 0
for i in range(len(input_training)):
    expected = output_training[i]
    activations, _, _ = forward(input_training[i], weights, biases)
    predicted = activations[-1]
    error += np.sum(np.subtract(expected, predicted) ** 2, axis=0)
    if np.nanargmax(predicted) == np.nanargmax(expected):
        correct += 1

# return results
print("")
print(f"Results - Loss: {round(error[0] / len(input_training), 5)} - Elapsed Time: {round(end_time - start_time, 5)}s - Accuracy: {round(correct / len(input_training) * 100, 5)}%")

# save results
if save:
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

# matplotlib graphs
if generate_graphs:
    y_true = []
    y_pred = []
    for i in range(len(input_training)):
        activations, _, _ = forward(input_training[i], weights, biases)
        y_true.append(np.nanargmax(activations[-1]))
        y_pred.append(np.nanargmax(output_training[i]))
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plot_cm(cm, title="Neural Network Results", labels=output_index)

    x_stuff = np.array(saved_epochs)
    y_stuff = np.array(saved_errors)
    plot_graph(data=[x_stuff, y_stuff], title="Error vs Epoch", labels=["Epoch", "Error"])


# network application
while True:
    print("")
    # get inputs
    inputs = []
    for input_node in range(layer_sizes[0]):
        inputs.append(float(input(f"{input_index[input_node]}: ")))

    # forward pass
    inputs = np.reshape(inputs, (len(inputs), 1))
    activations, _, _ = forward(inputs, weights, biases)

    # result
    print(activations[-1])
    print(f"Outputted: {output_index[np.nanargmax(activations[-1])]}")
