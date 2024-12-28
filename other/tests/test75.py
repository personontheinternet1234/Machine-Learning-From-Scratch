import numpy as np
from gardenpy.utils.objects_2 import Tensor2 as Tensor

tens1 = Tensor(np.random.randn(5, 5))
tens2 = Tensor(np.random.randn(5, 5))
print(tens2.id)
print(Tensor._instances)
tens2 = tens2.to_array()
print(Tensor._instances)
Tensor.reset_tracker()
