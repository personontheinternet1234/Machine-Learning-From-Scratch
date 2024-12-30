import numpy as np
from gardenpy.utils.objects_2 import Tensor
from gardenpy.utils.algorithms import Initializers, Activators

g = Activators('lrelu').activate
init = Initializers('xavier').initialize

tens1 = init(5, 5)
tens2 = init(5, 5)
tens3 = g(tens1 * tens2)

grad1 = Tensor.nabla(tens3, tens1)
print(grad1)
