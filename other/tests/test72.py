import numpy as np

from gardenpy import Tensor, nabla, chain

# tensor instantiation
t1 = Tensor(np.random.randn(1, 5))
t2 = Tensor(np.random.randn(1, 5))
t3 = Tensor(np.random.randn(1, 5))

# print tensors
print(t1)
print(t2)
print(t3)

# example calculations and tracking
r1 = t1 * t2
r2 = r1 * t3
print('individual gradients')
print(nabla(r1, t2))
print(nabla(r2, r1))
print('chain-ruled gradients')
print(nabla(r2, t2))
print(chain(nabla(r2, r1), nabla(r1, t2)))

# if I introduce a tensor not related to any of the calculations, an error appears
u1 = Tensor(np.random.randn(1, 5))
print(nabla(t1, u1))

