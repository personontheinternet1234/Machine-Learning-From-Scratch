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

# X = [np.array([[0, 1]]), np.array([[1, 1]]), np.array([[1, 0]]), np.array([[0, 0]])]
# Y = [np.array([[1, 0]]), np.array([[0, 1]]), np.array([[1, 0]]), np.array([[0, 1]])]
X = [np.array([[0, 0]]), np.array([[0, 1]]), np.array([[1, 0]]), np.array([[1, 1]])]
Y = [np.array([[1, 1]]), np.array([[1, 0]]), np.array([[0, 1]]), np.array([[0, 0]])]

W0 = np.random.randn(ins, hls)
B0 = np.zeros((1, hls))
W1 = np.random.randn(hls, ots)
B1 = np.zeros((1, ots))

Ll = []
il = []
testa = []

# f
for i in range(1000):
    # f a
    A0 = X
    A1 = l_relu(np.matmul(A0, W0) + B0)
    A2 = l_relu(np.matmul(A1, W1) + B1)
    E = Y

    # b a
    # l2
    dA2 = -2 * (E - A2)
    dB1 = dA2
    dW1 = np.reshape(A1, (4, 3, 1)) * dB1

    # l1
    dA1 = np.reshape(np.array([np.sum(W1 * dB1, axis=2)]), (4, 1, 3))
    dB0 = dA1
    dW0 = np.reshape(A0, (4, 2, 1)) * dB0

    # o
    W0 = W0 - lr * np.sum(dW0, axis=0) / len(X)
    B0 = B0 - lr * np.sum(dB0, axis=0) / len(X)
    W1 = W1 - lr * np.sum(dW1, axis=0) / len(X)
    B1 = B1 - lr * np.sum(dB1, axis=0) / len(X)

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

while True:
    print("")
    inputs = []
    for i in range(ins):
        inputs.append(float(input(f"{str(i)}: ")))

    # f
    A0_s = np.array(inputs)
    A1_s = l_relu(np.matmul(A0_s, W0) + B0)
    A2_s = l_relu(np.matmul(A1_s, W1) + B1)

    # r
    print("")
    print(A2_s)
