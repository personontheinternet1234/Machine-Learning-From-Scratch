import ast
import random
import time

import numpy as np
# import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

"""
This program uses the nodes structure to practice basic backpropagation.
Made from scratch (No tutorials, no pytorch).
Version: 1.0
Author: Isaac Park Verbrugge, Christian Host-Madsen
"""


# sigmoid activator
def sigmoid(values):
    output = 1 / (1 + np.exp(-1 * values))
    return output


# derivative of sigmoid activator
def d_sigmoid(values):
    # output = 1 / (1 + np.exp(-1 * values)) * (1 - 1 / (1 + np.exp(-1 * values)))
    output = sigmoid(values) * (1 - sigmoid(values))
    return output


# leaky rectified linear activator
def l_relu(values):
    output = np.maximum(0.1 * values, values)
    return output


# derivative of leaky rectified linear activator
def d_l_relu(values):
    return np.where(values > 0, 1, 0.1)


# # softmax activator
def softmax(values):
    return np.exp(values) / np.sum(np.exp(values))


# cross entropy error function
def centropy(softmax_probs, true_labels):
    true_label_index = np.where(true_labels > 0)[0][0]
    return -np.log(softmax_probs[true_label_index])


# derivative of cross entropy error function
def d_centropy(values, true_labels):
    # derivative is just softmax, unless you are the winner, then it is softmax - 1
    true_label_index = np.where(true_labels > 0)[0][0]
    softmax_probs = softmax(values)
    d_loss_d_values = softmax_probs.copy()
    d_loss_d_values[true_label_index] -= 1
    return d_loss_d_values


# list reformatting
def vectorize(list):
    vector = np.reshape(np.array(list), (len(list), 1))
    return vector


# xavier initialization
def xavier_initialize(length, width):
    array = np.random.randn(length, width) * np.sqrt(2 / length)
    return array


# zero initialized array
def zeros_initialize(length, width):
    array = np.zeros((length, width))
    return array


# forward pass
def forward(inputs, weights, biases):
    activations = [inputs]
    for layer in range(layers - 1):
        activation = l_relu(np.matmul(weights[layer], activations[-1]) + biases[layer])
        activations.append(activation)
    return activations, weights, biases


# backpropagation
def backward(activations, weights, biases, predicted):
    # initialize lists
    d_activations = []
    d_weights = []
    d_biases = []

    # error with respect to last layer
    d_activations.insert(0, -2 * np.subtract(predicted, activations[-1]))

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
        # weights[layer] = np.subtract(weights[layer], learning_rate * d_weights[layer])
        weights[layer] = np.subtract(weights[layer], learning_rate * (d_weights[layer] + (lambda_reg / train_len) * weights[layer]))
        biases[layer] = np.subtract(biases[layer], learning_rate * d_biases[layer])

    # return activations, weights, biases
    return weights, biases


def backward_2(activations, weights, biases, expected):
    # initialize gradient lists
    d_activations = []
    d_weights = []
    d_biases = []
    layers = len(activations)

    # calculate gradients
    d_activations.insert(0, np.multiply(-2, np.subtract(expected, activations[-1])))
    for connection in range(-1, -layers + 1, -1):
        d_biases.insert(0, d_l_relu(np.matmul(weights[connection], activations[connection - 1]) + biases[connection]) * d_activations[0])
        d_weights.insert(0, np.multiply(np.resize(d_biases[0], (layer_sizes[connection - 1], layer_sizes[connection])).T, np.resize(activations[connection - 1], (layer_sizes[connection], layer_sizes[connection - 1]))))
        d_activations.insert(0, np.reshape(np.sum(np.multiply(np.resize(d_biases[0], (layer_sizes[connection - 1], layer_sizes[connection])), weights[connection].T), axis=1), (layer_sizes[connection - 1], 1)))
    d_biases.insert(0, d_l_relu(np.matmul(weights[0], activations[0]) + biases[0]) * d_activations[0])
    d_weights.insert(0, np.multiply(np.resize(d_biases[0], (layer_sizes[0], layer_sizes[1])).T, np.resize(activations[0], (layer_sizes[1], layer_sizes[0]))))

    # apply gradients
    weights_output = []
    biases_output = []
    weights_size = len(weights)
    biases_size = len(biases)
    for connection in range(weights_size):
        weights_output.append(np.subtract(weights[connection], learning_rate * (d_weights[connection] + (lambda_reg / train_len) * weights[connection])))
        # weights_output.append(np.subtract(weights[connection], learning_rate * d_weights[connection]))
    for connection in range(biases_size):
        biases_output.append(np.subtract(biases[connection], learning_rate * d_biases[connection]))
    return weights_output, biases_output

# graph confusion matrix
def plot_cm(cm, title=None, labels=None, color="binary"):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )
    disp.plot(cmap=color)
    disp.ax_.set_title(title)
    plt.show()


# graph normal graph
def plot_graph(data, title=None, labels=None, color="black"):
    plt.plot(data[0], data[1], color=color)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)
    plt.show()


# split training and testing data
def test_train_split(data, test_size=0.3):
    random.shuffle(data)
    test = data[0:round(len(data) * test_size)]
    train = data[round(len(data) * test_size):]
    return train, test


# network settings
learn = True
load = False
save = False
graphs = True
epochs = 1000000
return_rate = 1
learning_rate = 0.0001
lambda_reg = 0.1

# network structure
layer_sizes = [2, 3, 2]

# dataset
X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
Y = [
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1]
]

# user indexes
X_labels = ["a(0)0", "a(0)1"]
Y_labels = ["checkered", "non checkered"]

# preset values
layers = len(layer_sizes)
train_len = len(X)

""" network code """

# reformat training data
for i in range(train_len):
    X[i] = vectorize(X[i])
for i in range(train_len):
    Y[i] = vectorize(Y[i])

start_time = time.time()

# instantiate weights and biases
weights = []
biases = []
if load:
    # load weights and biases
    with open("etc/weights.txt", "r") as file:
        weights = ast.literal_eval(file.read())
    with open("etc/biases.txt", "r") as file:
        biases = ast.literal_eval(file.read())
    for i in range(len(weights)):
        weights[i] = np.array(weights[i])
    for i in range(len(biases)):
        biases[i] = np.array(biases[i])
else:
    # generate weights and biases
    for i in range(layers - 1):
        weights.append(xavier_initialize(layer_sizes[i + 1], layer_sizes[i]))
        biases.append(np.zeros((layer_sizes[i + 1], 1)))

# network training
saved_epochs = []
saved_errors = []
if learn:
    # training loop
    for epoch in range(epochs):
        # SGD choice
        training_choice = int(np.random.rand() * len(X))
        inputs = X[training_choice]
        predicted = Y[training_choice]

        # forward pass
        activations, weights, biases = forward(inputs, weights, biases)

        # backpropagation
        weights, biases = backward(activations, weights, biases, predicted)

        # loss calculation
        if epoch % return_rate == 0:
            # SSR
            loss = 0
            for i in range(len(X)):
                activations, _, _ = forward(X[i], weights, biases)
                loss += np.sum(np.subtract(Y[i], activations[-1]) ** 2)
            saved_epochs.append(epoch)
            saved_errors.append(loss)
            # print(f"({round((epoch / epochs) * 100)}%) MSE: {error / len(input_training)}")

end_time = time.time()

""" return results """

# calculate accuracies
loss = 0
correct = 0
for i in range(train_len):
    expected = Y[i]
    activations, _, _ = forward(X[i], weights, biases)
    predicted = activations[-1]
    loss += np.sum(np.subtract(expected, predicted) ** 2)
    if np.nanargmax(predicted) == np.nanargmax(expected):
        correct += 1

# return results
print("")
print(f"Results - Loss: {round(loss / len(X), 5)} - Elapsed Time: {round(end_time - start_time, 5)}s - Accuracy: {round(correct / len(X) * 100, 5)}%")

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
if graphs:
    y_true = []
    y_pred = []
    for i in range(train_len):
        activations, _, _ = forward(X[i], weights, biases)
        y_true.append(np.nanargmax(activations[-1]))
        y_pred.append(np.nanargmax(Y[i]))
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plot_cm(cm, title="Neural Network Results", labels=Y_labels)

    plot_graph([np.array(saved_epochs), np.array(saved_errors)], title="Loss v.s. Epoch", labels=["Epoch", "Loss"])

# network application
while True:
    print("")
    # get inputs
    inputs = []
    for input_node in range(layer_sizes[0]):
        inputs.append(float(input(f"{X_labels[input_node]}: ")))

    # forward pass
    inputs = np.reshape(inputs, (len(inputs), 1))
    activations, _, _ = forward(inputs, weights, biases)

    # result
    print(activations[-1])
    print(f"Outputted: {Y_labels[np.nanargmax(activations[-1])]}")
