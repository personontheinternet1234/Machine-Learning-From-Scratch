import numpy as np


# sigmoid activator
def sigmoid(values):
    output = 1 / (1 + np.exp(-1 * values))
    return output


# derivative of sigmoid activator
def d_sigmoid(values):
    # output = 1 / (1 + np.exp(-1 * values)) * (1 - 1 / (1 + np.exp(-1 * values)))
    output = sigmoid(values) * (1 - sigmoid(values))
    return output


# softmax activator
def softmax(values):
    return np.exp(values) / np.sum(np.exp(values))


# cross entropy error function
def centropy(softmax_probs, true_labels):
    true_label_index = np.where(true_labels > 0)[0][0]
    return -np.log(softmax_probs[true_label_index])


# derivative of cross entropy error function
def d_centropy(values, true_labels):
    # derivative is just softmax, unless you are the winner, then it is softmax - 1
    true_label_index = np.where(true_labels > 0)[0][0]
    softmax_probs = softmax(values)
    d_loss_d_values = softmax_probs.copy()
    d_loss_d_values[true_label_index] -= 1
    return d_loss_d_values
