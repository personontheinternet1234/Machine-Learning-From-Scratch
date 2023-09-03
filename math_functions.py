import numpy as np

# Learning rate. Multiplied to result of gradient descent for new weight
learningRate = 0.1

def relU(x):  # Rectified linear activation function
    return max(x,0)

def calculate(incoming_energy, weight, bias):  # simple calculate function to find the activation energy of a new node
    return (incoming_energy * weight + bias)


# def in_out(polynomial, x_value): # takes a polynomial (hopefully we can graph the neural network's guesses as y with a given input as x) and calculates y for x
#     ...

# def SSR(observedDataSetCurveFormula, actualDataSet):  # sum of squared values function
#     for value in actualDataSet:
#         sum += ( in_out(observedDataSetCurveFormula, value[0])  - value[1] ) ** 2 # for each x value

# TODO: observedDataSet
def new_value(observedDataSet, actualDataSet, oldWeight):  # gradient descent function for a given connection's weight.
    # take derivitive of sum of squared values with respect to w4:
    # sum of each data point: 2 * (observed - predicted) * -1
    sum = 0
    for weight in actualDataSet:
        sum += -2 * (observedDataSet - actualDataSet)


    # returns next weight or bias the connection should have
    return oldWeight - (sum * learningRate)


if __name__ == "__main__":
    ...
