"""
A Neural Network made using only mathematical functions
Isaac Park Verbrugge & Christian SW Host-Madsen
"""

import ast
import random
import time

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

import keras  # keras's number dataset

""" definitions """


# split training and testing data
def test_train_split(data, test_size):
    random.shuffle(data)
    test = data[0:round(len(data) * test_size)]
    train = data[round(len(data) * test_size):]
    return train, test


# xavier initialization
def xavier_initialize(length, width):
    array = np.random.randn(length, width) * np.sqrt(2 / length)
    return array


# leaky rectified linear activator
def l_relu(values):
    output = np.maximum(0.1 * values, values)
    return output


# derivative of leaky rectified linear activator
def d_l_relu(values):
    output = np.where(values > 0, 1, 0.1)
    return output


# forward pass
def forward(inputs, weights, biases):
    nodes = [inputs]
    for layer in range(layers - 1):
        activations = l_relu(np.matmul(nodes[-1], weights[layer]) + biases[layer])
        nodes.append(activations)
    return nodes


# sgd backpropagation
def sgdbackward(nodes, expected, weights, biases):
    # initialize gradient lists
    d_weights = []
    d_biases = []

    # d_b = -2 * (expected - nodes[-1])
    # d_biases.insert(0, d_b)
    # d_biases.append(d_b)
    d_biases.insert(0, -2 * (expected - nodes[-1]))
    for layer in range(-1, -len(nodes) + 1, -1):
        # d_w = nodes[layer - 1].T * d_b
        # d_weights.insert(0, d_w)
        # d_weights.append(d_w)
        d_weights.insert(0, nodes[layer - 1].T * d_biases[0])
        # d_b = np.array([np.sum(weights[layer] * d_b, axis=1)])
        # d_biases.insert(0, d_b)
        # d_biases.append(d_b)
        d_biases.insert(0, np.array([np.sum(weights[layer] * d_biases[0], axis=1)]))
    # d_w = nodes[0].T * d_b
    # d_weights.insert(0, d_w)
    # d_weights.append(d_w)
    d_weights.insert(0, nodes[0].T * d_biases[0])

    # for layer in range(len(nodes) - 1):
    #     weights[layer] -= learning_rate * (d_weights[-layer - 1] + (lambda_reg / train_len) * weights[layer])
    #     biases[layer] -= learning_rate * d_biases[-layer - 1]

    for layer in range(len(nodes) - 1):
        weights[layer] -= learning_rate * (d_weights[layer] + (lambda_reg / train_len) * weights[layer])
        biases[layer] -= learning_rate * d_biases[layer]

    return weights, biases


# tensor backpropagation
def tensorbackward(nodes, expected, weights, biases):
    # initialize gradient lists
    d_weights = []
    d_biases = []

    d_b = -2 * (expected - nodes[-1])
    d_biases.insert(0, d_b)
    for layer in range(-1, -len(nodes) + 1, -1):
        # shape_a = np.shape(nodes[layer - 1])
        # d_w = np.reshape(nodes[layer - 1], (shape_a[0], shape_a[2], 1)) * d_b
        d_w = np.reshape(nodes[layer - 1], (train_len, layer_sizes[layer - 1], 1)) * d_b
        d_weights.insert(0, d_w)
        # d_b = np.array([np.sum(weights[layer] * d_b, axis=1)])
        # shape_c = np.shape(np.array([np.sum(weights[layer] * d_b, axis=2)]))
        # d_b = np.reshape(np.array([np.sum(weights[layer] * d_b, axis=2)]), (shape_c[1], 1, shape_c[2]))
        d_b = np.reshape(np.array([np.sum(weights[layer] * d_b, axis=2)]), (train_len, 1, layer_sizes[layer - 1]))
        d_biases.insert(0, d_b)
    # shape_a = np.shape(nodes[0])
    # d_w = np.reshape(nodes[0], (shape_a[0], shape_a[2], 1)) * d_b
    d_w = np.reshape(nodes[0], (train_len, layer_sizes[0], 1)) * d_b
    d_weights.insert(0, d_w)

    for layer in range(len(nodes) - 1):
        weights[layer] -= learning_rate * np.sum((d_weights[layer] + (lambda_reg / train_len) * weights[layer]), axis=0) / train_len
        biases[layer] -= learning_rate * np.sum(d_biases[layer], axis=0) / train_len

    return weights, biases


# graph confusion matrix
def plot_cm(cm, title=None, labels=None, color="Blues"):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )
    disp.plot(cmap=color)
    disp.ax_.set_title(title)
    plt.show()


# network params
learn = True
sgd = False
load = False
save = False
graphs = True
layer_sizes = [784, 16, 16, 10]
epochs = 1000
learning_rate = 0.001
lambda_reg = 0.1
log_rate = 1

# dataset params
trim = True
trim_value = 7000

# user indexes
Y_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

""" mnist """

# load data from mnist
(train_values, train_labels), (test_values, test_labels) = keras.datasets.mnist.load_data()
# combine training and testing labels
import_data_values = np.append(train_values, test_values, axis=0)
import_data_labels = np.append(train_labels, test_labels, axis=0)
data_values = []
data_labels = []

# reformat data from mnist
for i in range(len(train_values)):
    data_values.append(np.array([np.divide(import_data_values[i].flatten().tolist(), 255)]))
    node_values = np.zeros(layer_sizes[-1])
    node_values[train_labels[i]] = 1
    node_values = np.array([node_values])
    data_labels.append(node_values)

""" mnist end """

# trim dataset
if trim:
    data_values = data_values[0:trim_value]
    data_labels = data_labels[0:trim_value]
# split training and testing data
train, test = test_train_split(list(zip(data_values, data_labels)), test_size=0.3)
# unzip training and testing data
X, Y = zip(*train)
X_test, Y_test = zip(*test)
# reformat training and testing data
X, Y = list(X), list(Y)
X_test, Y_test = list(X_test), list(Y_test)

# network values
layers = len(layer_sizes)
train_len = len(X)
test_len = len(X_test)

""" network code """

# instantiate weights and biases
weights = []
biases = []
if load:
    # load weights and biases
    with open("etc/weights.txt", "r") as f:
        for line in f:
            weights.append(np.array(ast.literal_eval(line)))
    with open("etc/biases.txt", "r") as f:
        for line in f:
            biases.append(ast.literal_eval(line))
else:
    # generate weights and biases
    for i in range(layers - 1):
        weights.append(xavier_initialize(layer_sizes[i], layer_sizes[i + 1]))
        biases.append(np.zeros((1, layer_sizes[i + 1])))

# network training
start_time = time.time()

logged_epochs = []
logged_losses = []
logged_losses_test = []
if learn:
    # training loop
    for epoch in tqdm(range(epochs), ncols=100):
        if sgd:
            # SGD choice
            training_choice = int(np.random.rand() * len(X))
            inputs = X[training_choice]
            expected = Y[training_choice]

            # forward pass
            nodes = forward(inputs, weights, biases)

            # backpropagation
            weights, biases = sgdbackward(nodes, expected, weights, biases)

            # loss calculation
            if epoch % log_rate == 0:
                # SSR
                train_predicted = forward(X, weights, biases)[-1]
                test_predicted = forward(X_test, weights, biases)[-1]
                loss = np.sum(np.subtract(Y, train_predicted) ** 2) / train_len
                test_loss = np.sum(np.subtract(Y_test, test_predicted) ** 2) / test_len
                logged_epochs.append(epoch)
                logged_losses.append(loss)
                logged_losses_test.append(test_loss)
        else:
            # tensors
            inputs = X
            expected = Y

            # forward pass
            nodes = forward(inputs, weights, biases)

            # backpropagation
            weights, biases = tensorbackward(nodes, expected, weights, biases)

            # loss calculation
            if epoch % log_rate == 0:
                # SSR
                train_predicted = forward(X, weights, biases)[-1]
                test_predicted = forward(X_test, weights, biases)[-1]
                loss = np.sum(np.subtract(Y, train_predicted) ** 2) / train_len
                test_loss = np.sum(np.subtract(Y_test, test_predicted) ** 2) / test_len
                logged_epochs.append(epoch)
                logged_losses.append(loss)
                logged_losses_test.append(test_loss)

end_time = time.time()

""" return results """

# train loss
train_predicted = forward(X, weights, biases)[-1]
loss = np.sum(np.subtract(Y, train_predicted) ** 2) / train_len
# test loss
test_predicted = forward(X_test, weights, biases)[-1]
loss_test = np.sum(np.subtract(Y_test, test_predicted) ** 2) / test_len

# train accu
accu = 0
for i in range(len(X)):
    predicted = forward(X[i], weights, biases)[-1]
    if np.nanargmax(predicted) == np.nanargmax(Y[i]):
        accu += 1
accu /= train_len

# test accu
accu_test = 0
for i in range(len(X_test)):
    predicted = forward(X_test[i], weights, biases)[-1]
    if np.nanargmax(predicted) == np.nanargmax(Y_test[i]):
        accu_test += 1
accu_test /= test_len

# print results
print("")
print(f"Results - Train Loss: {round(loss, 5)} - Test Loss: {round(loss_test, 5)} - Train Accuracy: {round(accu, 5)} - Test Accuracy: {round(accu_test, 5)} - Elapsed Time: {round(end_time - start_time, 5)}s")

# save optimized weights and biases
if save:
    with open("etc/weights.txt", "w") as f:
        for array in range(len(weights)):
            f.write(str(weights[array].tolist()) + "\n")
    with open("etc/biases.txt", "w") as f:
        for array in range(len(biases)):
            f.write(str(biases[array].tolist()) + "\n")

# matplotlib graphs
if graphs:
    # generate cms
    # train
    y_true = []
    y_pred = []
    for i in range(len(X)):
        predicted = forward(X[i], weights, biases)[-1]
        expected = Y[i]
        y_true.append(np.nanargmax(predicted))
        y_pred.append(np.nanargmax(expected))
    cm_train = confusion_matrix(y_true, y_pred, normalize="true")

    # test
    y_true_test = []
    y_pred_test = []
    for i in range(len(X_test)):
        predicted = forward(X_test[i], weights, biases)[-1]
        expected = Y_test[i]
        y_true_test.append(np.nanargmax(predicted))
        y_pred_test.append(np.nanargmax(expected))
    cm_test = confusion_matrix(y_true_test, y_pred_test, normalize="true")

    # generate loss vs epoch
    logged_epochs = np.array(logged_epochs)
    logged_losses = np.array(logged_losses)
    logged_losses_test = np.array(logged_losses_test)

    # graph cms
    plot_cm(cm_train, title="Train Results", labels=Y_names)
    plot_cm(cm_test, title="Test Results", labels=Y_names)

    # graph loss vs epoch
    plt.plot(logged_epochs, logged_losses, color="blue", label="Train")
    plt.plot(logged_epochs, logged_losses_test, color="red", label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss v.s. Epoch")
    plt.legend(loc="lower right")
    plt.show()
