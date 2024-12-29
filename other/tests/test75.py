import numpy as np
from gardenpy.utils.objects_2 import Tensor

tens1 = Tensor(np.abs(np.random.randn(5, 5)))
tens2 = Tensor(np.abs(np.random.randn(5, 5)))
tens3 = tens1 * tens2
