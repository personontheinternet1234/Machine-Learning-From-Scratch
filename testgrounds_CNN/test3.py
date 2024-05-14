import random

import numpy as np

from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

import keras

""" definitions """


def relu(values):
    output = np.maximum(0, values)
    return output


def sigmoid(values):
    output = 1 / (1 + np.exp(-1 * values))
    return output


# s/o ChatGPT
def max_pooling(feature_map, pool_size=(2, 2), strides=(2, 2)):
    h_out = (feature_map.shape[0] - pool_size[0]) // strides[0] + 1
    w_out = (feature_map.shape[1] - pool_size[1]) // strides[1] + 1
    pooled_feature_map = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            patch = feature_map[i * strides[0]:i * strides[0] + pool_size[0],
                    j * strides[1]:j * strides[1] + pool_size[1]]
            pooled_feature_map[i, j] = np.max(patch)
    return pooled_feature_map


def l_relu(values):
    output = np.maximum(0.1 * values, values)
    return output


def d_l_relu(values):
    return np.where(values > 0, 1, 0.1)


def plot_cm(cm, title=None, labels=None, color="Blues"):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )
    disp.plot(cmap=color)
    disp.ax_.set_title(title)
    plt.show()


""" superparams """

ins = 100
ots = 10
eps = 100000
lr = 0.001

""" image data """

# load data from mnist
(X_temp, Y_temp), (X_test, Y_test) = keras.datasets.mnist.load_data()
X = np.append(X_temp, X_test, axis=0)
Y_temp_2 = np.append(Y_temp, Y_test, axis=0)

Y = []
for i in tqdm(range(len(X)), ncols=150, desc="Reformatting Data"):
    n_v = np.zeros(10)
    n_v[Y_temp_2[i]] = 1
    n_v = np.array([n_v])
    Y.append(n_v)

""" instantiated values """

W0 = np.random.randn(ins, ots)
B0 = np.zeros((1, ots))

# layer 1 kernels
kernel_1_1_1 = np.array([[0.003, -0.077, -0.037, 0.039, -0.095], [0.103, -0.345, 0.220, 0.010, 0.137], [0.028, -0.259, -0.108, -0.077, 0.140], [-0.078, -0.062, 0.021, -0.015, -0.018], [-0.042, 0.067, 0.088, 0.051, 0.200]])
kernel_1_1_2 = np.array([[0.028, 0.060, 0.046, 0.069, 0.214], [-0.093, -0.019, 0.036, 0.230, -0.115], [-0.265, -0.430, -0.266, -0.278, -0.328], [-0.005, -0.029, -0.244, -0.132, -0.216], [0.111, -0.035, 0.197, 0.183, 0.136]])

# layer 2 kernels
kernel_1_2_1 = np.array([[-0.442, -0.124, -0.237], [-0.121, -0.207, -0.136], [0.104, 0.040, 0.207]])
kernel_1_2_2 = np.array([[0.100, 0.040, 0.067], [-0.174, 0.050, 0.063], [-0.025, -0.073, -0.248]])
kernel_1_2_3 = np.array([[-0.121, -0.114, -0.109], [0.238, 0.273, 0.007], [-0.065, 0.000, 0.102]])
kernel_1_2_4 = np.array([[-0.022, -0.030, 0.072], [0.113, 0.095, 0.037], [0.067, -0.006, 0.172]])

kernel_2_2_1 = np.array([[0.200, 0.011, 0.163], [0.019, -0.027, 0.178], [-0.172, 0.068, 0.037]])
kernel_2_2_2 = np.array([[-0.075, 0.217, 0.155], [-0.134, 0.068, 0.114], [0.024, 0.155, 0.180]])
kernel_2_2_3 = np.array([[-0.036, -0.167, -0.040], [0.064, -0.025, -0.094], [0.478, 0.113, -0.089]])
kernel_2_2_4 = np.array([[-0.016, 0.099, -0.688], [0.005, -0.157, -0.238], [-0.061, -0.130, -0.186]])

""" training """
Ll = []
for i in tqdm(range(eps), ncols=150, desc="Training"):
    # test choice
    tc = random.randint(0, len(X) - 1)
    img = X[tc]

    # layer 1 convolutions
    conv_img_1_1_1 = convolve2d(img, kernel_1_1_1, mode="valid")
    conv_img_1_1_1 = relu(conv_img_1_1_1)
    conv_img_1_1_1 = max_pooling(conv_img_1_1_1)

    conv_img_1_1_2 = convolve2d(img, kernel_1_1_2, mode="valid")
    conv_img_1_1_2 = relu(conv_img_1_1_2)
    conv_img_1_1_2 = max_pooling(conv_img_1_1_2)

    # layer 2 convolutions
    conv_img_1_3 = (convolve2d(conv_img_1_1_1, kernel_1_2_1, mode="valid") + convolve2d(conv_img_1_1_2, kernel_2_2_1, mode="valid")) / 2
    conv_img_1_3 = sigmoid(conv_img_1_3)
    conv_img_1_3 = max_pooling(conv_img_1_3)

    conv_img_2_3 = (convolve2d(conv_img_1_1_1, kernel_1_2_2, mode="valid") + convolve2d(conv_img_1_1_2, kernel_2_2_2, mode="valid")) / 2
    conv_img_2_3 = sigmoid(conv_img_2_3)
    conv_img_2_3 = max_pooling(conv_img_2_3)

    conv_img_3_3 = (convolve2d(conv_img_1_1_1, kernel_1_2_3, mode="valid") + convolve2d(conv_img_1_1_2, kernel_2_2_3, mode="valid")) / 2
    conv_img_3_3 = sigmoid(conv_img_3_3)
    conv_img_3_3 = max_pooling(conv_img_3_3)

    conv_img_4_3 = (convolve2d(conv_img_1_1_1, kernel_1_2_4, mode="valid") + convolve2d(conv_img_1_1_2, kernel_2_2_4, mode="valid")) / 2
    conv_img_4_3 = sigmoid(conv_img_4_3)
    conv_img_4_3 = max_pooling(conv_img_4_3)

    # final layer reformatting
    conv_img_12_3 = np.append(conv_img_1_3, conv_img_2_3)
    conv_img_34_3 = np.append(conv_img_3_3, conv_img_4_3)
    conv_img_1234_3 = np.array([np.append(conv_img_12_3, conv_img_34_3)])

    # fully connected network
    # forward
    A0 = conv_img_1234_3
    A1 = l_relu(np.matmul(A0, W0) + B0)
    E = Y[tc]

    # backward
    dA1 = -2 * (E - A1)
    dB0 = dA1
    dW0 = A0.T * dB0

    # optimize (i can't believe i forgot to do this what am i doing)
    W0 = W0 - lr * dW0
    B0 = B0 - lr * dB0

    # loss
    L = np.sum(np.subtract(E, A1) ** 2)
    Ll.append(L)

plt.plot(range(eps), Ll, color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss v.s. Epoch")
plt.show()

y_true = []
y_pred = []
for i in tqdm(range(len(X)), ncols=150, desc="Generating CM"):
    img = X[i]

    # layer 1 convolutions
    conv_img_1_1_1 = convolve2d(img, kernel_1_1_1, mode="valid")
    conv_img_1_1_1 = relu(conv_img_1_1_1)
    conv_img_1_1_1 = max_pooling(conv_img_1_1_1)

    conv_img_1_1_2 = convolve2d(img, kernel_1_1_2, mode="valid")
    conv_img_1_1_2 = relu(conv_img_1_1_2)
    conv_img_1_1_2 = max_pooling(conv_img_1_1_2)

    # layer 2 convolutions
    conv_img_1_3 = (convolve2d(conv_img_1_1_1, kernel_1_2_1, mode="valid") + convolve2d(conv_img_1_1_2, kernel_2_2_1,
                                                                                        mode="valid")) / 2
    conv_img_1_3 = sigmoid(conv_img_1_3)
    conv_img_1_3 = max_pooling(conv_img_1_3)

    conv_img_2_3 = (convolve2d(conv_img_1_1_1, kernel_1_2_2, mode="valid") + convolve2d(conv_img_1_1_2, kernel_2_2_2,
                                                                                        mode="valid")) / 2
    conv_img_2_3 = sigmoid(conv_img_2_3)
    conv_img_2_3 = max_pooling(conv_img_2_3)

    conv_img_3_3 = (convolve2d(conv_img_1_1_1, kernel_1_2_3, mode="valid") + convolve2d(conv_img_1_1_2, kernel_2_2_3,
                                                                                        mode="valid")) / 2
    conv_img_3_3 = sigmoid(conv_img_3_3)
    conv_img_3_3 = max_pooling(conv_img_3_3)

    conv_img_4_3 = (convolve2d(conv_img_1_1_1, kernel_1_2_4, mode="valid") + convolve2d(conv_img_1_1_2, kernel_2_2_4,
                                                                                        mode="valid")) / 2
    conv_img_4_3 = sigmoid(conv_img_4_3)
    conv_img_4_3 = max_pooling(conv_img_4_3)

    # final layer reformatting
    conv_img_12_3 = np.append(conv_img_1_3, conv_img_2_3)
    conv_img_34_3 = np.append(conv_img_3_3, conv_img_4_3)
    conv_img_1234_3 = np.array([np.append(conv_img_12_3, conv_img_34_3)])

    # fully connected network
    # forward
    A0 = conv_img_1234_3
    A1 = l_relu(np.matmul(A0, W0) + B0)
    E = Y[i]
    y_true.append(np.nanargmax(A1))
    y_pred.append(np.nanargmax(E))

cm = confusion_matrix(y_true, y_pred, normalize="true")
plot_cm(cm, title="CNN Train Results")
