import numpy as np


def cost(name):
    """ get raw cost function """
    j = {
        'mse': lambda y, yhat: np.sum((y - yhat) ** 2),
        'l1': lambda y, yhat: np.sum(np.abs(y - yhat)),
        'cross-entropy': lambda y, yhat: np.exp(yhat[0][y if isinstance(y, int) else np.nanargmax(y)]) / np.sum(np.exp(yhat))
    }
    if name in j:
        return j[name]
    else:
        raise ValueError(f"'{name}' is an invalid loss function")


cost_func = cost('cross-entropy')
expected = np.zeros((1, 10))
expected[0][5] = 1
predicted = np.abs(np.random.randn(1, 10))

loss = cost_func(expected, predicted)
expected = 5
loss_2 = cost_func(expected, predicted)

print(loss)
print(loss_2)
