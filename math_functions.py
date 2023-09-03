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
#         sum += ( inout(observedDataSetCurveFormula, value[0])  - value[1] ) ** 2 # for each x value

# TODO: observedDataSet



if __name__ == "__main__":
    ...
