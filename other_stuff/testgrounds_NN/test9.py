import time

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
ep = 10000
ins = 2
hls = 3
ots = 2

X = np.array([np.array([[0, 0]]), np.array([[0, 1]]), np.array([[1, 0]]), np.array([[1, 1]])])
Y = np.array([np.array([[1, 1]]), np.array([[1, 0]]), np.array([[0, 1]]), np.array([[0, 0]])])

W0 = np.random.randn(ins, hls)
B0 = np.zeros((1, hls))
W1 = np.random.randn(hls, ots)
B1 = np.zeros((1, ots))

Ll = []
t1 = time.time()
for i in tqdm(range(ep), ncols=150):
    # f
    A0 = X
    A1 = l_relu(np.matmul(A0, W0) + B0)
    A2 = l_relu(np.matmul(A1, W1) + B1)
    E = Y

    # b
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
    L = np.sum(np.subtract(Y, A2) ** 2) / len(X)
    Ll.append(L)
t2 = time.time()

print(t2 - t1)
plt.plot(range(ep), Ll, color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss v.s. Epoch")
plt.show()
