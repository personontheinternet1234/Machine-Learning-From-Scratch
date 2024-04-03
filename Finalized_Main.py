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
def vectorize(values):
    vector = np.reshape(np.array(values), (len(values), 1))
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
    return activations


# backpropagation
def backward(activations, predicted, weights, biases):
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
        weights[layer] = np.subtract(weights[layer],
                                     learning_rate * (d_weights[layer] + (lambda_reg / train_len) * weights[layer]))
        biases[layer] = np.subtract(biases[layer], learning_rate * d_biases[layer])

    # return activations, weights, biases
    return weights, biases


def backward_2(activations, expected, weights, biases):
    # initialize gradient lists
    d_activations = []
    d_weights = []
    d_biases = []
    layers = len(activations)

    # calculate gradients
    d_activations.insert(0, np.multiply(-2, np.subtract(expected, activations[-1])))
    for layer in range(-1, -layers + 1, -1):
        d_biases.insert(0,
                        d_l_relu(np.matmul(weights[layer], activations[layer - 1]) + biases[layer]) * d_activations[0])
        d_weights.insert(0, np.multiply(np.resize(d_biases[0], (layer_sizes[layer - 1], layer_sizes[layer])).T,
                                        np.resize(activations[layer - 1],
                                                  (layer_sizes[layer], layer_sizes[layer - 1]))))
        d_activations.insert(0, np.reshape(
            np.sum(np.multiply(np.resize(d_biases[0], (layer_sizes[layer - 1], layer_sizes[layer])), weights[layer].T),
                   axis=1), (layer_sizes[layer - 1], 1)))
    d_biases.insert(0, d_l_relu(np.matmul(weights[0], activations[0]) + biases[0]) * d_activations[0])
    d_weights.insert(0, np.multiply(np.resize(d_biases[0], (layer_sizes[0], layer_sizes[1])).T,
                                    np.resize(activations[0], (layer_sizes[1], layer_sizes[0]))))

    # apply gradients
    weights_output = []
    biases_output = []
    weights_size = len(weights)
    biases_size = len(biases)
    for connection in range(weights_size):
        weights_output.append(np.subtract(weights[connection], learning_rate * (
                    d_weights[connection] + (lambda_reg / train_len) * weights[connection])))
        # weights_output.append(np.subtract(weights[connection], learning_rate * d_weights[connection]))
    for connection in range(biases_size):
        biases_output.append(np.subtract(biases[connection], learning_rate * d_biases[connection]))
    return weights_output, biases_output


# graph confusion matrix
def plot_cm(cm, title=None, labels=None, color="Blues"):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )
    disp.plot(cmap=color)
    disp.ax_.set_title(title)
    plt.show()


# split training and testing data
def test_train_split(data, test_size):
    random.shuffle(data)
    test = data[0:round(len(data) * test_size)]
    train = data[round(len(data) * test_size):]
    return train, test


# network settings
learn = True
load = False
save = False
graphs = True
epochs = 10000
log_rate = 1
learning_rate = 0.01
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

X = X * 5
Y = Y * 5

# user indexes
X_labels = ["a(0)0", "a(0)1"]
Y_labels = ["checkered", "non checkered"]

data_len = len(X)

# reformat data
for i in range(data_len):
    X[i] = vectorize(X[i])
for i in range(data_len):
    Y[i] = vectorize(Y[i])

# split training and testing data
train, test = test_train_split(list(zip(X, Y)), test_size=0.3)
# unzip training and testing data
if len(train) == 0:
    X = []
    Y = []
else:
    X, Y = zip(*train)
if len(test) == 0:
    X_test = []
    Y_test = []
else:
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
    for epoch in range(epochs):
        # SGD choice
        training_choice = int(np.random.rand() * len(X))
        inputs = X[training_choice]
        expected = Y[training_choice]

        # forward pass
        neurons = forward(inputs, weights, biases)

        # backpropagation
        weights, biases = backward_2(neurons, expected, weights, biases)

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
            loss = loss / train_len
            test_loss = test_loss / test_len
            logged_epochs.append(epoch)
            logged_losses.append(loss)
            logged_losses_test.append(test_loss)
            # print(f"({round((epoch / epochs) * 100)}%) MSE: {error / len(input_training)}")

end_time = time.time()

""" return results """

# calculate accuracies
loss = 0
correct = 0
for i in range(train_len):
    predicted = forward(X[i], weights, biases)[-1]
    expected = Y[i]
    loss += np.sum(np.subtract(expected, predicted) ** 2)
loss = loss / train_len
for i in range(test_len):
    predicted = forward(X_test[i], weights, biases)[-1]
    expected = Y_test[i]
    if np.nanargmax(predicted) == np.nanargmax(expected):
        correct += 1

# return results
print("")
print(f"Results - Train Loss: {round(loss, 5)} - Elapsed Time: {round(end_time - start_time, 5)}s - Test Accuracy: {round(correct / test_len * 100, 5)}%")

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
    # confusion matrix graph
    y_true = []
    y_pred = []
    for i in range(test_len):
        predicted = forward(X_test[i], weights, biases)[-1]
        expected = Y_test[i]
        y_true.append(np.nanargmax(predicted))
        y_pred.append(np.nanargmax(expected))
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plot_cm(cm, title="Test Results", labels=Y_labels)

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
        inputs.append(float(input(f"{X_labels[input_node]}: ")))

    # forward pass
    inputs = np.reshape(inputs, (len(inputs), 1))
    neurons = forward(inputs, weights, biases)
    predicted = neurons[-1]

    # result
    print(predicted)
    print(f"Predicted: {Y_labels[np.nanargmax(predicted)]}")
