from gardenpy import (
    nabla,
    Initializers
)

init = Initializers(algorithm='gaussian')
arr1 = init.initialize(3, 5)
arr2 = init.initialize(5, 4)

arr3 = arr1 @ arr2

print(nabla(arr3, arr2))
