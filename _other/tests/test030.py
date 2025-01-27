import numpy as np

from _other.deprecated.objects import Tensor
from _other.deprecated.operators import nabla, chain

t1 = Tensor(np.random.randn(3, 3))
t2 = Tensor(np.random.randn(3, 3))
t3 = Tensor(np.random.randn(3, 3))
t4 = Tensor(np.random.randn(3, 3))

c1 = t1 + t2
c2 = c1 - t3
c3 = c2 * t4
print(id(c1))
print(id(c2))
print(id(c3))

print()
print(repr(c2))

print()
d_c1t1 = nabla(c1, t1)
print(repr(d_c1t1))
d_c2c1 = nabla(c2, c1)
print(repr(d_c2c1))
d_c3c2 = nabla(c3, c2)
print(repr(d_c3c2))

print()

d_c3c1 = chain(d_c3c2, d_c2c1)
print(repr(d_c3c1))
d_c3t1 = chain(d_c3c1, d_c1t1)
print(repr(d_c3t1))
