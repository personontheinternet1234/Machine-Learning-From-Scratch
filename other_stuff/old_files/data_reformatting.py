from tensorflow import keras  # import dataset

# load dataset
X, Y = keras.datasets.mnist.load_data()

# save dataset to file
with open("saved/weights.txt", "w") as f:
    f.write(str(X))
    # for datapoint in X:
    #     f.write(str(datapoint))
with open("saved/biases.txt", "w") as f:
    f.write(str(Y))
    # for datapoint in Y:
    #     f.write(str(datapoint))

