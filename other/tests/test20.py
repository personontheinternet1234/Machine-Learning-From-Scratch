import random

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
it = 1
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

for i in range(it):
    # ch
    # tc = random.randint(0, 7)
    tc = 7

    # f
    a1 = x[tc]
    y1 = y[tc]
    a2 = g1(a1 @ w1 + b1)
    a3 = g1(a2 @ w2 + b2)
    c = j(a3, y1)

    a2f = g1(x @ w1) + b1
    a3f = g1(a2f @ w2) + b2
    cf = j(a3f, y)

    # print(a2)
    # print(a3)
    # print(c)
    # tb()
    # print(a2f)
    # print(a3f)
    # print(cf)
    # tb()

    # b
    # l3
    da3 = dj(a3, y1)
    # l2
    db2 = dg1(a2 @ w2 + b2) * da3
    dw2 = a2.T * db2
    da2 = np.sum(w2 * db2, axis=1)
    # l1
    db1 = dg1(a1 @ w1 + b1) * da2
    dw1 = a1.T * db1

    da3f = dj(a3f, y)
    db2f = dg1(a2f @ w2 + b2) * da3f
    dw2f = np.reshape(a2f, (8, l2, 1)) * db2f
    da2f = np.reshape(np.sum(w2 * db2f, axis=2, keepdims=True), (8, 1, l2))
    db1f = dg1(x @ w1 + b1) * da2f
    dw1f = np.reshape(x, (8, l1, 1)) * db1f

    def test_eq():
        return None
        # print(np.shape(a2f))
        # a2ft = []
        # for k in range(len(a2f)):
        #     a2ft.append(a2f[k].T)
        # a2ft = np.array(a2ft)
        # print(a2ft)
        # print(np.shape(a2ft))
        # tb()
        # print(a2f)
        # print(np.reshape(a2f, (8, l2, 1)))
        # print(np.reshape(a2f, (3, 1, 8)))

    # print(da3)
    # tb()
    # print(da3f)
    # tb()
    # print(db2)
    # tb()
    # print(db2f)
    # tb()
    # print(dw2)
    # tb()
    # print(dw2f)
    # tb()
    # print(da2)
    # tb()
    # print(da2f)
    # tb()
    # print(db1)
    # tb()
    # print(db1f)
    # tb()
    print(dw1)
    tb()
    print(dw1f)
    tb()

    # o
    b2 -= lr * db2
    w2 -= lr * dw2
    b1 -= lr * db1
    w1 -= lr * dw1

    # r
    # print(c)
    # tb()
