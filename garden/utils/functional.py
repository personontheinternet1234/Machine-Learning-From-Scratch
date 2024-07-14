"""
mathematical functions
"""

import numpy as np

# todo: deprecate
def ssr(expected, predicted):
    c = np.sum(np.subtract(expected, predicted) ** 2)
    return c


def xavier(length, width):
    # todo: deprecate
    """ initialize a random array using xavier initialization """
    return np.random.randn(length, width) * np.sqrt(2 / length)


def softmax(x):
    # todo: deprecate
    """ calculate outcome probabilities using softmax """
    return np.exp(x) / np.sum(np.exp(x))


def initializers(name, gain='auto'):
    if gain == 'auto':
        gains = {
            'xavier': 1
        }
        if name in gains:
            gain = gains[name]
        else:
            gain = None
    init = {
        'gaussian': lambda l, w: np.random.randn(l, w),
        'xavier': lambda l, w: np.random.randn(l, w) * gain / np.sqrt(2 / (w + l)),
        'zeros': lambda l, w: np.zeros((l, w)),
        'ones': lambda l, w: np.ones((l, w))
    }
    if name in init:
        return init[name]
    else:
        raise ValueError(f"'{name}' is an invalid initialization method")


def activators(name, beta='auto'):
    """ get raw activation function """
    # set beta
    if beta == 'auto':
        betas = {
            'leaky relu': 0.01,
            'softplus': 1,
            'mish': 1
        }
        if name in betas:
            beta = betas[name]
        else:
            beta = None
    elif isinstance(beta, float):
        beta = beta
    else:
        raise ValueError(f"'beta' ({beta}) must be a float or 'auto'")
    # set activation function
    g = {
        'softmax': lambda x: np.exp(x) / np.sum(np.exp(x)),
        'relu': lambda x: np.maximum(0, x),
        'leaky-relu': lambda x: np.maximum(beta * x, x),
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'tanh': lambda x: np.tanh(x),
        'softplus': lambda x: np.log(1 + np.exp(beta * x)) / beta,
        'mish': lambda x: x * np.tanh(np.log(1 + np.exp(beta * x)) / beta)
    }
    # return activation function
    if name in g:
        return g[name]
    else:
        raise ValueError(f"'{name}' is an invalid activation function")


def d_activators(name, beta='auto'):
    """ get derivative of raw activation function """
    # set beta
    if beta == 'auto':
        betas = {
            'leaky relu': 0.01,
            'softplus': 1,
            'mish': 1
        }
        if name in betas:
            beta = betas[name]
        else:
            beta = 0
    elif isinstance(beta, float):
        beta = beta
    else:
        raise ValueError(f"'beta' ({beta}) must be a float or 'auto'")
    dg = {
        'softmax': lambda x: x,  # todo
        'relu': lambda x: np.where(x > 0, 1, 0),
        'leaky-relu': lambda x: np.where(x > 0, 1, beta),
        'sigmoid': lambda x: np.exp(-x) / ((1 + np.exp(-x)) ** 2),
        'tanh': lambda x: np.cosh(x) ** -2,
        'softplus': lambda x: beta * np.exp(beta * x) / (beta + beta * np.exp(beta * x)),
        'mish': lambda x: (np.tanh(np.log(1 + np.exp(beta * x)) / beta)) + (x * (np.cosh(np.log(1 + np.exp(beta * x)) / beta) ** -2) * (beta * np.exp(beta * x) / (beta + beta * np.exp(beta * x))))
    }
    if name in dg:
        return dg[name]
    else:
        raise ValueError(f"'{name}' is an invalid activation function")


def losses(name):
    """ get raw cost function """
    j = {
        'ssr': lambda y, yhat: np.sum((y - yhat) ** 2),
        'sar': lambda y, yhat: np.sum(np.abs(y - yhat)),
        'centropy': lambda y, yhat: np.sum(yhat * np.log(y))
    }
    if name in j:
        return j[name]
    else:
        raise ValueError(f"'{name}' is an invalid loss function")


def d_losses(name):
    """ get derivative of raw cost function """
    dj = {
        'ssr': lambda y, yhat: -2 * (y - yhat),
        'sar': lambda y, yhat: -1 * np.sign(y - yhat),
        'centropy': lambda y, yhat: ...  # todo
    }
    if name in dj:
        return dj[name]
    else:
        raise ValueError(f"'{name}' is an invalid loss function")


class Optimizers:
    def __init__(self, name, hyperparameters='auto', store=True):
        # optimizer parameters
        self._name = name
        self._hyperparams = self._get_hyperparams(hyperparameters)
        self._optim = self._get_optim(name)

        # memory parameters
        self._storage = self._get_storage(store)
        self._deltas_p = None
        self._upsilons_p = None
        self._upsilons_ms = None

    @staticmethod
    def _get_hyperparams(hyperparams):
        # define auto hyperparams
        auto = {
            'gammas': {
                'adam': 0.001,
                'sgd': 0.001,
                'rms': 0.001
            },
            'betas': {
                'adam': (0.9, 0.999),
                'rms': 0.99,
            },
            'lambda_ds': {
                'adam': 0,
                'sgd': 0,
                'rms': 0
            },
            'mus': {
                'sgd': 0,
                'rms': 0
            },
            'taus': {
                'sgd': 0
            },
            'epsilons': {
                'adam': 1e-8,
                'rms': 1e-8
            },
            'ams': {
                'adam': False
            },
            'nesterov': {
                'sgd': False
            }
        }
        if hyperparams == 'auto':
            # set hyperparams if auto
            hyperparams = auto
        elif isinstance(hyperparams, dict):
            # set hyperparams if defined
            hyperparams = auto
            UserWarning('hyperparameters have to be auto at this time')  # todo: allow changing of hyperparameters
        else:
            # errors
            raise TypeError(
                f"hyperparameters must be a dictionary\n"
                f"(hyperparameters: {hyperparams})"
            )
        # return hyperparams
        return hyperparams

    @staticmethod
    def _get_optim(method):
        # todo: check order of operations
        def adam(thetas, gradients, deltas_p=None, upsilons_p=None, upsilons_mx=None, gamma=0.001, lambda_d=0, beta=(0.9, 0.999), epsilon=1e-8, ams=False):
            # weight decay
            deltas = gradients + (lambda_d * thetas)
            if deltas_p:
                # momentum
                deltas = ((beta[0] * deltas_p) + ((1 - beta[0]) * deltas)) / (1 - beta[0])
            if upsilons_p:
                # square momentum
                upsilons = ((beta[1] * upsilons_p) / (1 - beta[1])) + (deltas ** 2)
            else:
                # square
                upsilons = deltas ** 2
            if ams:
                # ams-grad variant
                # todo: check math here
                if upsilons_mx:
                    upsilons_mx = np.maximum(upsilons_mx, upsilons)
                else:
                    upsilons_mx = upsilons
                upsilons = upsilons_mx
            # optimization
            thetas -= gamma * deltas / (np.sqrt(upsilons) + epsilon)
            # param return
            return thetas, deltas, upsilons, upsilons_mx

        def sgd(thetas, gradients, deltas_p=None, gamma=0.001, lambda_d=0, mu=0, tau=0, nesterov=False):
            # weight decay
            deltas = gradients + (lambda_d * thetas)
            if mu and deltas_p:
                # momentum
                deltas = (mu * deltas_p) + ((1 - tau) * deltas)
            if nesterov and mu:
                # nesterov momentum
                deltas += gradients + mu * deltas
            # optimization
            thetas -= gamma * deltas
            # param return
            return thetas, deltas

        def rms(thetas, gradients, deltas_p=None, upsilons_p=None, gamma=0.01, lambda_d=0, beta=0.99, mu=0, epsilon=1e-8):
            # weight decay
            deltas = gradients + (lambda_d * thetas)
            if upsilons_p:
                # square momentum
                upsilons = (beta * upsilons_p) + ((1 - beta) * (deltas ** 2))
            else:
                # square
                upsilons = (1 - beta) * (deltas ** 2)
            # step calculation
            deltas = deltas / (np.sqrt(upsilons) + epsilon)
            if mu and deltas_p:
                # momentum
                deltas += mu * deltas_p
            # optimization
            thetas -= gamma * deltas
            # param return
            return thetas, deltas, upsilons

        methods = {
            'adam': adam,
            'sgd': sgd,
            'rms': rms
        }

        if method in methods:
            return methods[method]
        else:
            raise ValueError(
                f"'{method}' is an invalid optimization technique\n"
                f"choose from {[name for name in methods]}"
            )

    def _get_storage(self, store):
        storing = {
            'deltas_p': ['adam', 'sgd', 'rms'],
            'upsilons_p': ['adam, rms'],
            'upsilons_ms': ['adam']
        }
        if store:
            stores = []
            for param, method in storing:
                if self.name in storing[param]:
                    stores.append(param)
            return stores
        else:
            return None

    def set


def optimizers(name, parameters='auto'):
    # todo: check how well python/numpy respects the order of operations
    hyperparams = {
        'gammas': {
            'adam': 0.001,
            'sgd': 0.001,
            'rms': 0.001
        },
        'betas': {
            'adam': (0.9, 0.999),
            'rms': 0.99,
        },
        'lambda_ds': {
            'adam': 0,
            'sgd': 0,
            'rms': 0
        },
        'mus': {
            'sgd': 0,
            'rms': 0
        },
        'taus': {
            'sgd': 0
        },
        'epsilons': {
            'adam': 1e-8,
            'rms': 1e-8
        },
        'ams': {
            'adam': False
        },
        'nesterov': {
            'sgd': False
        }
    }

    def adam(thetas, gradients, deltas_p=None, upsilons_p=None, upsilons_mx=None, gamma=0.001, lambda_d=0, beta=(0.9, 0.999), epsilon=1e-8, ams=False):
        # weight decay
        deltas = gradients + (lambda_d * thetas)
        if deltas_p:
            # momentum
            deltas = ((beta[0] * deltas_p) + ((1 - beta[0]) * deltas)) / (1 - beta[0])
        if upsilons_p:
            # square momentum
            upsilons = ((beta[1] * upsilons_p) / (1 - beta[1])) + (deltas ** 2)
        else:
            # square
            upsilons = deltas ** 2
        if ams:
            # ams-grad variant
            upsilons_mx = np.maximum(upsilons_mx, upsilons)
            upsilons = upsilons_mx
        # optimization
        thetas -= gamma * deltas / (np.sqrt(upsilons) + epsilon)
        # param return
        return thetas, deltas, upsilons, upsilons_mx

    def sgd(thetas, gradients, deltas_p=None, gamma=0.001, lambda_d=0, mu=0, tau=0, nesterov=False):
        # weight decay
        deltas = gradients + (lambda_d * thetas)
        if mu and deltas_p:
            # momentum
            deltas = (mu * deltas_p) + ((1 - tau) * deltas)
        if nesterov and mu:
            # nesterov momentum
            deltas += gradients + mu * deltas
        # optimization
        thetas -= gamma * deltas
        # param return
        return thetas, deltas

    def rms(thetas, gradients, upsilons_p=None, deltas_p=None, gamma=0.01, lambda_d=0, beta=0.99, mu=0, epsilon=1e-8):
        # weight decay
        deltas = gradients + (lambda_d * thetas)
        if upsilons_p:
            # square momentum
            upsilons = (beta * upsilons_p) + ((1 - beta) * (deltas ** 2))
        else:
            # square
            upsilons = (1 - beta) * (deltas ** 2)
        # step calculation
        deltas = deltas / (np.sqrt(upsilons) + epsilon)
        if mu and deltas_p:
            # momentum
            deltas += mu * deltas_p
        # optimization
        thetas -= gamma * deltas
        # param return
        return thetas, deltas, upsilons
    methods = {
        'adam': adam,
        'sgd': sgd,
        'rms': rms
    }
    if name in methods:
        return methods[name]
    else:
        raise ValueError(
            f"'{name}' is an invalid optimization technique\n"
            f"choose from {[method for method in methods]}"
        )
