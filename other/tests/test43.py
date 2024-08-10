from gardenpy import (
    nabla,
    Initializers,
    Activators
)

init = Initializers(algorithm='xavier')
act = Activators(algorithm='relu')

arr1 = init.initialize(1, 5)
arr2 = act.activate(arr1)
arr3 = act.activate(arr2)

print(arr1)
print(arr2)
print(arr3)

print('----------')
d1 = nabla(arr2, arr1)
d2 = nabla(arr3, arr2)
d3 = nabla(arr3, arr1)
print(d1)
print(d2)
print(d3)
