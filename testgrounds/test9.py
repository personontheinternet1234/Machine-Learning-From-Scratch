import random
import time

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from tqdm import tqdm


def l_relu(values):
    output = tf.maximum(0.1 * values, values)
    return output


def d_l_relu(values):
    return tf.where(values > 0, 1, 0.1)


def tb():
    print("---------------")


lr = 0.01
ep = 10000
ins = 2
hls = 3
ots = 2

X = [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]
Y = [[[1, 1]], [[1, 0]], [[0, 1]], [[0, 0]]]
X = tf.constant(X, dtype=tf.float32)
Y = tf.constant(Y, dtype=tf.float32)

W0 = tf.Variable(tf.random.uniform((ins, hls)))
B0 = tf.Variable(tf.zeros((1, hls)))
W1 = tf.Variable(tf.random.uniform((hls, ots)))
B1 = tf.Variable(tf.zeros((1, ots)))

Ll = []
t1 = time.time()
for i in tqdm(range(ep), ncols=150):
    # f
    A0 = X
    A1 = l_relu(tf.matmul(A0, W0) + B0)
    A2 = l_relu(tf.matmul(A1, W1) + B1)
    E = Y

    # b
    # l2
    dA2 = -2 * (E - A2)
    dB1 = dA2
    dW1 = tf.reshape(A1, [4, 3, 1]) * dB1

    # l1
    dA1 = tf.reshape(tf.reduce_sum(W1 * dB1, axis=2), [4, 1, 3])
    dB0 = dA1
    dW0 = tf.reshape(A0, [4, 2, 1]) * dB0

    # o
    W0 = W0 - lr * tf.reduce_sum(dW0, axis=0) / len(X)
    B0 = B0 - lr * tf.reduce_sum(dB0, axis=0) / len(X)
    W1 = W1 - lr * tf.reduce_sum(dW1, axis=0) / len(X)
    B1 = B1 - lr * tf.reduce_sum(dB1, axis=0) / len(X)

    # e
    X1 = l_relu(tf.matmul(X, W0) + B0)
    X2 = l_relu(tf.matmul(X1, W1) + B1)
    L = tf.reduce_sum(tf.subtract(Y, X2) ** 2) / len(X)

    # s
    Ll.append(L)
t2 = time.time()

print(t2 - t1)
plt.plot(range(ep), Ll, color="blue")
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
