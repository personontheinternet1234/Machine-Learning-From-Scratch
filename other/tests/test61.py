import numpy as np

from gardenpy import Initializers, nabla

init = Initializers('gaussian').initialize

t1 = init(4, 3)
t2 = init(3, 1)
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

