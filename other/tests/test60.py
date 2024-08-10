import numpy as np

from gardenpy import Tensor, nabla, chain, Activators

act1 = Activators('softmax').activate

tens1 = Tensor(np.random.randn(5, 5))
tens2 = Tensor(np.random.randn(5, 5))

res1 = act1(tens1 * tens2)

print(nabla(res1, tens1).get_internals())
