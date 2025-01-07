import numpy as np


def _get_solver(self):
    if self._batching == 1:
        def backward(a, y):
            # instantiate gradients
            grad_w = self._zero_grad[0]
            grad_b = self._zero_grad[1]

            # calculate gradients
            grad_a = self._j['d'](y, a[-1])
            for lyr in range(-1, -len(a) + 1, -1):
                grad_b[lyr] = self._g[lyr]['d'](a[lyr - 1] @ self._w[lyr] + self._b[lyr]) * grad_a
                grad_w[lyr] = a[lyr - 1].T * grad_b[lyr]
                grad_a = np.sum(self._w[lyr] * grad_b[lyr], axis=1)
            grad_b[0] = self._g[0]['d'](a[0] @ self._b[0] + self._b[0]) * grad_a
            grad_w[0] = a[0].T * grad_b[0]

            # return gradients
            return grad_w, grad_b

        # return solver
        return backward

    else:
        # redefine zero gradients
        self._zero_grad[0] = np.array([self._zero_grad[0]] * self._batching)
        self._zero_grad[0] = np.array([self._zero_grad[1]] * self._batching)

        def update(a, y):
            # instantiate gradients
            grad_w = self._zero_grad[0]
            grad_b = self._zero_grad[1]

            # calculate gradients
            grad_a = self._j['d'](y, a[-1])
            for lyr in range(-1, -len(a) + 1, -1):
                grad_b[lyr] = self._g[lyr]['d'](a[lyr - 1] @ self._w[lyr] + self._b[lyr]) * grad_a
                grad_w[lyr] = np.reshape(a[lyr - 1], (self._batching, self._lyrs[lyr - 1], 1)) * grad_b
                grad_a = np.reshape(np.sum(self._w[lyr] * grad_b[lyr], axis=2, keepdims=True),
                                    (self._batching, 1, self._lyrs[lyr - 1]))
            grad_b[0] = self._g[0]['d'](a[0] @ self._w[0] + self._b[0]) * grad_a
            grad_w[0] = np.reshape(a[0], (self._batching, self._lyrs[0], 1)) * grad_b[0]

            # sum and average gradients
            grad_w = np.sum(grad_w, axis=0) / self._batching
            grad_b = np.sum(grad_b, axis=0) / self._batching

            # return gradients
            return grad_w, grad_b

        # return solver
        return update
