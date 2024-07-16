import numpy as np

from gardenpy.utils.objects import Tensor
from gardenpy.utils.operations import nabla

x = Tensor(np.random.randn(1, 5))
w = Tensor(np.random.randn(5, 5))
# w = np.random.randn(5, 5)
b = Tensor(np.ones((1, 5)))
y = Tensor(np.random.randn(1, 5))

# print(x)
# print(w)

a_t = x @ w
y_hat = a_t + b

print(id(b))

grad_b = nabla(y_hat, b)
print(repr(grad_b))

# grad_a_t = nabla(y_hat, a_t)
# print(grad_a_t)
#
# grad_w = nabla(a_t, w)
# print(grad_w)
#
# grad_x = nabla(a_t, x)
# print(grad_x)
