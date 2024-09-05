import numpy as np

from gardenpy import Tensor, nabla, chain, Activators, Losses

act = Activators('sigmoid').activate
loss = Losses('ssr').loss

t1 = Tensor(np.random.randn(1, 5))
t2 = Tensor(np.random.randn(1, 5))

t3 = t1 + t2
t4 = act(t3)

r1 = Tensor(np.random.randn(1, 5))
c = loss(t4, r1)

print(nabla(t3, t1))
print(nabla(t4, t3))
print(nabla(t4, t1))
print(chain(nabla(t4, t3), nabla(t3, t1)))
print()

print(nabla(c, t1))
print(chain(nabla(c, t4), nabla(t4, t1)))
print(nabla(c, t4))
