import random

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def l_relu(values):
    output = np.maximum(0.1 * values, values)
    return output


def d_l_relu(values):
    return np.where(values > 0, 1, 0.1)


def tb():
    print("---------------")


lr = 0.01
ins = 2
hls = 3
ots = 2

X = [np.array([[0, 0]]), np.array([[0, 1]]), np.array([[1, 0]]), np.array([[1, 1]])]
Y = [np.array([[1, 1]]), np.array([[1, 0]]), np.array([[0, 1]]), np.array([[0, 0]])]

W0 = np.random.randn(ins, hls)
B0 = np.zeros((1, hls))
W1 = np.random.randn(hls, ots)
B1 = np.zeros((1, ots))

Ll = []
il = []

# f
for i in tqdm(range(10000)):
    # f a
    A0 = X
    A1 = l_relu(np.matmul(A0, W0) + B0)
    A2 = l_relu(np.matmul(A1, W1) + B1)

    # b a
    # l2
    dA2 = -2 * (E - A2_s)
    dB1 = dA2
    dW1 = A1_s.T * dB1
    # l1
    dA1 = np.array([np.sum(W1 * dB1, axis=1)])
    dB0 = dA1
    dW0 = A0_s.T * dB0

    dW0ao = dW0
    dB0ao = dB0
    dW1ao = dW1
    dB1ao = dB1

    dW0a = 0
    dB0a = 0
    dW1a = 0
    dB1a = 0
    for tc in range(len(X)):
        # f
        A0_s = X[tc]
        A1_s = l_relu(np.matmul(A0_s, W0) + B0)
        A2_s = l_relu(np.matmul(A1_s, W1) + B1)
        E = Y[tc]

        # b
        # l2
        dA2 = -2 * (E - A2_s)
        dB1 = dA2
        dW1 = A1_s.T * dB1
        # l1
        dA1 = np.array([np.sum(W1 * dB1, axis=1)])
        dB0 = dA1
        dW0 = A0_s.T * dB0

        dW0a += dW0
        dB0a += dB0
        dW1a += dW1
        dB1a += dB1

    # optimize
    W0 = W0 - lr * dW0a / len(X)
    B0 = B0 - lr * dB0a / len(X)
    W1 = W1 - lr * dW1a / len(X)
    B1 = B1 - lr * dB1a / len(X)

    # e
    X1 = l_relu(np.matmul(X, W0) + B0)
    X2 = l_relu(np.matmul(X1, W1) + B1)
    L = np.sum(np.subtract(Y, X2) ** 2) / len(X)

    # s
    il.append(i)
    Ll.append(L)

plt.plot(np.array(il), np.array(Ll), color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss v.s. Epoch")
plt.show()
