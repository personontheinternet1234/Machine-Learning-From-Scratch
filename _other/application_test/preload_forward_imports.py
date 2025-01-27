# with gardenpy imports

def main(a0=None):
    import ast
    import os

    import numpy as np
    from gardenpy import Activators

    def load_params(root):
        weights = []
        biases = []
        with open(os.path.join(root, 'weights.txt'), 'r') as f:
            for line in f:
                weights.append(np.array(ast.literal_eval(line)))
        with open(os.path.join(root, 'biases.txt'), 'r') as f:
            for line in f:
                biases.append(np.array(ast.literal_eval(line)))
        return weights, biases

    act = Activators('lrelu', beta=0.1).activate
    pred = Activators('softmax').activate

    w, b = load_params(os.path.dirname(__file__))

    a1 = act(np.array([a0]) @ w[0] + b[0])
    a2 = act(a1 @ w[1] + b[1])
    a3 = act(a2 @ w[2] + b[2])

    return {'norm': a3, 'pred': np.argmax(a3), 'prob': pred(a3)}


if __name__ == '__main__':
    # todo: set input image here
    image = ...
    main(image)
