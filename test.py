import numpy as np

layers = 4
for layer in range(layers - 2, -1, -1):
    print(layer)

# list1 = [[1,2,3],
#          [4,5,6]]
# upstream = np.resize(list1, (2, 3)).T,
# upstream2 = np.resize(list1, (3, 2)),
# print(upstream)
# print(upstream2)