def backward(activations, predicted, weights, biases):
    # initialize lists
    d_activations = []
    d_weights = []
    d_biases = []

    # error with respect to last layer
    d_activations.insert(0, -2 * np.subtract(predicted, activations[-1]))

    # loop through layers backwards
    for layer in range(layers - 2, -1, -1):
        # gradient of biases
        d_b = d_l_relu(np.matmul(weights[layer], activations[layer]) + biases[layer]) * d_activations[0]
        d_biases.insert(0, d_b)

        # gradient of weights
        upstream = np.resize(d_biases[0], (len(activations[layer]), len(activations[layer + 1]))).T
        local = np.resize(activations[layer].T, (len(activations[layer + 1]), len(activations[layer])))

        d_w = np.multiply(upstream, local)
        d_weights.insert(0, d_w)

        # gradient of activations
        upstream = np.resize(d_biases[0], (len(activations[layer]), len(activations[layer + 1])))
        totals = np.sum(np.multiply(upstream, weights[layer].T), axis=1)

        d_a = np.reshape(totals, (len(activations[layer]), 1))
        d_activations.insert(0, d_a)

    for layer in range(layers - 2, -1, -1):
        # weights[layer] = np.subtract(weights[layer], learning_rate * d_weights[layer])
        weights[layer] = np.subtract(weights[layer],
                                     learning_rate * (d_weights[layer] + (lambda_reg / train_len) * weights[layer]))
        biases[layer] = np.subtract(biases[layer], learning_rate * d_biases[layer])

    # return activations, weights, biases
    return weights, biases