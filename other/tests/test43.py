import numpy as np

from gardenpy import (
    Tensor,
    nabla,
    chain,
    Initializers,
    Activators,
    Losses
)

a1 = Tensor(np.random.randn(1, 5))
y = np.random.randn(1, 5)
g = Activators('softmax')
j = Losses('centropy')
a2 = g.activate(a1)
a3 = j.loss(y, a2)

da3a2 = nabla(a3, a2)
da2a1 = nabla(a2, a1)
print(da3a2)
print(da2a1)
print(chain(da3a2, da2a1))
print(nabla(a3, a1))
