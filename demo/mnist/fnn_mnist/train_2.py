import numpy as np

from gardenpy.models import FNN

model = FNN()

model.initialize()

arr1 = np.random.randn(5)
print(np.nanargmax(arr1))
print(np.argmax(arr1))
