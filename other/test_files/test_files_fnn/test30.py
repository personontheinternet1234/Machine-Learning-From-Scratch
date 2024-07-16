import numpy as np

from gardenpy.utils.objects import Tensor
from gardenpy.utils.operations import nabla, chain

t1 = Tensor(np.random.randn(3, 3))
t2 = Tensor(np.random.randn(3, 3))
t3 = Tensor(np.random.randn(3, 3))
t4 = Tensor(np.random.randn(3, 3))

c1 = t1 + t2
c2 = c1 + t3
c3 = c2 + t4

d_c1t1 = nabla(c1, t1)
print(repr(d_c1t1))
d_c2c1 = nabla(c2, c1)
print(repr(d_c2c1))
d_c3c2 = nabla(c3, c2)
print(repr(d_c3c2))

print()
d_c2t1 = chain(d_c2c1, d_c1t1)
print(repr(d_c2t1))
d_c3t1 = chain(d_c3c2, d_c2t1)
print(repr(d_c3t1))
