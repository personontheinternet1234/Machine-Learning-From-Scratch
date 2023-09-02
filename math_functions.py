# Learning rate. Multiplied to result of gradient descent for new weight
learningRate = 0.1


def relU(x):  # Rectified linear activation function
    return max(x,0)

def calculate(incoming_energy, weight, bias):  # simple calculate function to find the activation energy of a new node
    return (incoming_energy * weight + bias)


def in_out(polynomial, x_value): # takes a polynomial (hopefully we can graph the neural network's guesses as y with a given input as x) and calculates y for x
        sum = 0
        for i in range(len(polynomial.powers)):
            sum += polynomial.coefficients[i] * (x_value ** function.powers[i])
        return sum

# actualDataSet = [
# [x0,y0],
# [x1,y1]
#
#
# ]

def SSR(observedDataSetCurveFormula, actualDataSet):  # sum of squared values function
    for value in actualDataSet:
        sum += ( in_out(observedDataSetCurveFormula, value[0])  - value[1] ) ** 2 # for each x value


def new_value(oldWeight):  # gradient descent function for a given connection's weight.
    # take derivitive of sum of squared values with respect to w4:
    # TODO
    derivitive = ...

    # returns next weight or bias the connection should have
    return oldWeight - (derivitive * learningRate)


if __name__ == "__main__":
    ...
