from gardenpy import utils
import os

loader = utils.dataloaders.DataLoaderCSV(os.getcwd(), "labels.csv", "values.csv")
loader.load()
print(loader._labels.get_internals())