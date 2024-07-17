import numpy as np

from gardenpy.utils.objects import Tensor
from gardenpy.utils.operations import nabla, chain, test_tracker

t1 = Tensor(np.random.randn(3, 3))
t2 = Tensor(np.random.randn(3, 3))
t3 = Tensor(np.random.randn(3, 3))
t4 = Tensor(np.random.randn(3, 3))

print(id(t1))
print(id(t2))
print(id(t3))
print(id(t4))
print()

c1 = t1 + t2
c2 = c1 - t3
c3 = c2 * t4
print(id(c1))
print(id(c2))
print(id(c3))

# print()
# print(repr(c1))
# print(repr(c2))
# print(repr(c3))

print()
print(test_tracker(c3, t1))
