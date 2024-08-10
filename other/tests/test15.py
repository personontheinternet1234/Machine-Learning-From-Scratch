import random
import time

import numpy as np


def g1(v):
    output = np.maximum(0.1 * v, v)
    return output


def dg1(v):
    return np.where(v > 0, 1, 0.1)


def g2(v):
    return np.exp(v) / np.sum(np.exp(v))


def dg2(v):
    # todo
    return v


def j(yh, yl):
    return np.sum((yl - yh) ** 2)


def dj(yh, yl):
    return -2 * (yl - yh)


def tb():
    print('----------')


# s
l1 = 3
l2 = 4
l3 = 3
it = 100000
lr = 0.1

# t
x = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
y = [[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
# fm
for i in range(len(x)):
    x[i] = [x[i]]
    y[i] = [y[i]]
x = np.array(x)
y = np.array(y)

# p
w1 = np.random.randn(l1, l2)
b1 = np.zeros((1, l2))
w2 = np.random.randn(l2, l3)
b2 = np.zeros((1, l3))

s = time.time()
for i in range(it):
    # ch
    tc = random.randint(0, 7)

    # f
    a1 = x[tc]
    y1 = y[tc]
    a2 = g1(a1 @ w1 + b1)
    a3 = g1(a2 @ w2 + b2)
    c = j(a3, y1)

    # b
    # l3
    da3 = dj(a3, y1)
    # l2
    db2 = dg1(a2 @ w2 + b2) * da3
    dw2 = (a2.T @ np.ones((1, l3))) * (np.ones((l2, 1)) @ db2)
    da2 = ((w2 * (np.ones((l2, 1)) @ db2)) @ np.ones((l1, 1))).T
    # l1
    db1 = dg1(a1 @ w1 + b1) * da2
    dw1 = (a1.T @ np.ones((1, l2))) * (np.ones((l1, 1)) @ db1)

    # o
    b2 -= lr * db2
    w2 -= lr * dw2
    b1 -= lr * db1
    w1 -= lr * dw1

    # r
    # print(c)
    # tb()

print(time.time() - s)
