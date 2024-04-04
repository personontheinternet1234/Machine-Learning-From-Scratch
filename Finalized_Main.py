import ast
import random
import time

import numpy as np
# import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from tensorflow import keras  # number training dataset

"""
This program uses the nodes structure to practice basic backpropagation.
Made from scratch (No tutorials, no pytorch).
Version: 1.0
Author: Isaac Park Verbrugge, Christian Host-Madsen
"""


""" unused definitions """


# sigmoid activator
def sigmoid(values):
    output = 1 / (1 + np.exp(-1 * values))
    return output


# derivative of sigmoid activator
def d_sigmoid(values):
    # output = 1 / (1 + np.exp(-1 * values)) * (1 - 1 / (1 + np.exp(-1 * values)))
    output = sigmoid(values) * (1 - sigmoid(values))
    return output


# softmax activator
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


""" used definitions """


# leaky rectified linear activator
def l_relu(values):
    output = np.maximum(0.1 * values, values)
    return output


# derivative of leaky rectified linear activator
def d_l_relu(values):
    return np.where(values > 0, 1, 0.1)


# turn list into vector
def vectorize(values):
    vector = np.reshape(np.array(values), (len(values), 1))
    return vector


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


# forward pass
def forward(inputs, weights, biases):
    nodes = [inputs]
    for layer in range(layers - 1):
        activation = l_relu(np.matmul(weights[layer], nodes[-1]) + biases[layer])
        nodes.append(activation)
    return nodes


# backpropagation
def backward(nodes, expected, weights, biases):
    # initialize gradient lists
    d_nodes = []
    d_weights = []
    d_biases = []

    # calculate gradients
    d_a = np.multiply(-2, np.subtract(expected, nodes[-1]))
    d_nodes.insert(0, d_a)
    for layer in range(-1, -len(nodes) + 1, -1):
        d_b = d_l_relu(np.matmul(weights[layer], nodes[layer - 1]) + biases[layer]) * d_nodes[0]
        d_biases.insert(0, d_b)
        d_w = np.multiply(np.resize(d_biases[0], (layer_sizes[layer - 1], layer_sizes[layer])).T, np.resize(nodes[layer - 1], (layer_sizes[layer], layer_sizes[layer - 1])))
        d_weights.insert(0, d_w)
        d_n = np.reshape(np.sum(np.multiply(np.resize(d_biases[0], (layer_sizes[layer - 1], layer_sizes[layer])), weights[layer].T), axis=1), (layer_sizes[layer - 1], 1))
        d_nodes.insert(0, d_n)
    d_b = d_l_relu(np.matmul(weights[0], nodes[0]) + biases[0]) * d_nodes[0]
    d_biases.insert(0, d_b)
    d_w = np.multiply(np.resize(d_biases[0], (layer_sizes[0], layer_sizes[1])).T, np.resize(nodes[0], (layer_sizes[1], layer_sizes[0])))
    d_weights.insert(0, d_w)

    # apply gradients
    for layer in range(len(nodes) - 1):
        weights[layer] -= learning_rate * (d_weights[layer] + (lambda_reg / train_len) * weights[layer])
    for layer in range(len(nodes) - 1):
        biases[layer] -= learning_rate * d_biases[layer]
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


# network settings
learn = False
load = True
save = False
graphs = True
epochs = 1000000
log_rate = 1000000
learning_rate = 0.001
lambda_reg = 0.1

# network structure
layer_sizes = [784, 16, 16, 10]

# user indexes
Y_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# set dataset

# I actually have 0 idea why this doesn't work
# X = []
# Y = []
# with open("data/data_input.txt", "r") as f:
#     for line in f:
#         X.append(ast.literal_eval(line))
# with open("data/data_input.txt", "r") as f:
#     for line in f:
#         Y.append(ast.literal_eval(line))

(train_values, train_labels), (test_values, test_labels) = keras.datasets.mnist.load_data()
X = []
Y = []

# reformat data
for i in range(len(train_values)):
    X.append(np.divide(train_values[i].flatten().tolist(), 255))
    X[i] = vectorize(X[i])
    node_values = np.zeros(10)
    node_values[train_labels[i]] = 1
    Y.append(vectorize(node_values))

# split training and testing data
train, test = test_train_split(list(zip(X, Y)), test_size=0.3)
# unzip training and testing data
X, Y = zip(*train)
X_test, Y_test = zip(*test)
# reformat training and testing data
X, Y = list(X), list(Y)
X_test, Y_test = list(X_test), list(Y_test)

# generated values
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
            biases.append(vectorize(ast.literal_eval(line)))
else:
    # generate weights and biases
    for i in range(layers - 1):
        weights.append(xavier_initialize(layer_sizes[i + 1], layer_sizes[i]))
        biases.append(np.zeros((layer_sizes[i + 1], 1)))

# network training
start_time = time.time()

logged_epochs = []
logged_losses = []
logged_losses_test = []
if learn:
    # training loop
    for epoch in tqdm(range(epochs), ncols=100):
        # SGD choice
        training_choice = int(np.random.rand() * len(X))
        inputs = X[training_choice]
        expected = Y[training_choice]

        # forward pass
        nodes = forward(inputs, weights, biases)

        # backpropagation
        weights, biases = backward(nodes, expected, weights, biases)

        # loss calculation
        if epoch % log_rate == 0:
            # SSR
            loss = 0
            test_loss = 0
            for i in range(len(X)):
                predicted = forward(X[i], weights, biases)[-1]
                loss += np.sum(np.subtract(Y[i], predicted) ** 2)
            for i in range(len(X_test)):
                predicted = forward(X_test[i], weights, biases)[-1]
                test_loss += np.sum(np.subtract(Y_test[i], predicted) ** 2)
            loss /= train_len
            test_loss /= test_len
            logged_epochs.append(epoch)
            logged_losses.append(loss)
            logged_losses_test.append(test_loss)

end_time = time.time()

""" return results """

# calculate results
loss = 0
correct = 0
for i in range(train_len):
    # SSR
    predicted = forward(X[i], weights, biases)[-1]
    loss += np.sum(np.subtract(Y[i], predicted) ** 2)
loss /= len(X)
for i in range(len(X_test)):
    # accuracies
    predicted = forward(X_test[i], weights, biases)[-1]
    if np.nanargmax(predicted) == np.nanargmax(Y_test[i]):
        correct += 1

# print results
print("")
print(f"Results - Train Loss: {round(loss, 5)} - Elapsed Time: {round(end_time - start_time, 5)}s - Test Accuracy: {round(correct / test_len * 100, 5)}%")

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
    # confusion matrix graph
    y_true = []
    y_pred = []
    for i in range(len(X_test)):
        predicted = forward(X_test[i], weights, biases)[-1]
        expected = Y_test[i]
        y_true.append(np.nanargmax(predicted))
        y_pred.append(np.nanargmax(expected))
    cm = confusion_matrix(y_true, y_pred, normalize="false")
    plot_cm(cm, title="Test Results", labels=Y_names)

    # loss vs epoch graph
    logged_epochs = np.array(logged_epochs)
    logged_losses = np.array(logged_losses)
    logged_losses_test = np.array(logged_losses_test)
    plt.plot(logged_epochs, logged_losses, color="blue", label="Train")
    plt.plot(logged_epochs, logged_losses_test, color="red", label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss v.s. Epoch")
    plt.legend(loc="lower right")
    plt.show()

# network application
while True:
    print("")
    # get inputs
    inputs = []
    for input_node in range(layer_sizes[0]):
        inputs.append(float(input(f"Node {input_node + 1}: ")))

    # forward pass
    inputs = np.reshape(inputs, (len(inputs), 1))
    nodes = forward(inputs, weights, biases)
    predicted = nodes[-1]

    # result
    print(predicted)
    print(f"Predicted: {Y_names[np.nanargmax(predicted)]}")
