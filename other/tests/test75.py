import numpy as np
from gardenpy.utils.operators_2 import tensor

tens1 = tensor(np.abs(np.random.randn(5, 5)))
tens2 = tensor(np.abs(np.random.randn(5, 5)))
tens3 = tens1 ** tens2
print(tens3.tracker)
print(tens1.tracker)
