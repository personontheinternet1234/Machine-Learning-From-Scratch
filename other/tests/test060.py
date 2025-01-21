import numpy as np

from gardenpy import Initializers, nabla

init = Initializers('gaussian').initialize

t1 = init(1, 3)
t2 = init(3, 4)
print(t1)
print(t2)
print('\n\n')

r1 = t1 @ t2
print(r1)
print('\n\n')

g1 = nabla(r1, t1)
g2 = nabla(r1, t2)

print(g1)
print(g2)
print('\n\n')

print(t2.to_array().T)
print(np.tile(t1.to_array().T, (1, t2.shape[1])))
