"""
'functional' includes mathematical algorithms for machine learning models.

'functional' includes:
    'Initializers': Initialization algorithms for kernels/weights/biases.
    'Activators': Activation algorithms for activations.
    'Losses': Loss algorithms for loss.
    'Optimizers': Optimization algorithms for kernels/weights/biases.

refer to '(todo)' for in-depth documentation for these algorithms.
"""

import numpy as np


class Initializers:
    def __init__(self, algorithm: str, parameters: dict = None):
        """
        'Initializers' is a class containing various initialization algorithms.
        These initialization algorithms currently include 'xavier' (Xavier/Glorot), 'gaussian' (Gaussian/Normal), 'zeros' (Uniform Zeros), and 'ones' (Uniform Ones).
        The class structure promotes flexibility through the easy addition of other initialization algorithms.

        Arguments:
            algorithm: A string referencing the desired initialization algorithm.
            parameters: A dictionary referencing the parameters for the initialization algorithm.
                (any parameters not referenced will be automatically defined)
        """
        # initialization algorithms
        self.algorithms = [
            'xavier',
            'gaussian',
            'zeros',
            'ones'
        ]

        # internal initialization algorithm parameters
        self._algorithm = self._check_algorithm(algorithm)
        self._params = self._get_params(parameters)

        # initialization algorithm
        self._initializer = self._get_initializer()

    def _check_algorithm(self, algorithm):
        # checks whether the initialization algorithm is valid
        if algorithm in self.algorithms:
            # valid initialization algorithm
            return algorithm
        else:
            # invalid initialization algorithm
            raise ValueError(
                f"Invalid initialization algorithm: '{algorithm}'\n"
                f"Choose from: {[alg for alg in self.algorithms]}"
            )

    def _get_params(self, params):
        # defines initialization algorithm parameters
        # default initialization algorithm parameters
        default = {
            'xavier': {
                'gain': 1
            },
            'gaussian': None,
            'zeros': None,
            'ones': None
        }

        # instantiate parameter dictionary
        prms = default[self._algorithm]

        if params and prms and isinstance(params, dict):
            # set defined parameter
            for prm in params:
                if prm in prms:
                    # todo: add errors for each hyperparameter
                    # valid parameter
                    prms[prm] = params[prm]
                else:
                    # invalid parameter
                    raise UserWarning(
                        f"Invalid parameter for '{self._algorithm}': {prm}\n"
                        f"Choose from: {[prm for prm in prms]}"
                    )
        elif params and isinstance(params, dict):
            # parameters not taken
            raise UserWarning(f"'{self._algorithm}' does not take parameters")
        elif params:
            # invalid data type
            raise TypeError(
                f"'parameters' is not a dictionary: {params}\n"
                f"Choose from: {[prm for prm in prms]}"
            )

        # return parameters
        return prms

    def _get_initializer(self):
        # defines initialization algorithm
        def gaussian(row, col):
            # Gaussian initialization
            return np.random.randn(row, col)

        def xavier(row, col):
            # Xavier/Glorot initialization
            return np.random.randn(row, col) * self._params['gain'] / np.sqrt(2 / (row + col))

        def zeros(row, col):
            # Zeros uniform initialization
            return np.zeros((row, col))

        def ones(row, col):
            # Ones uniform initialization
            return np.ones((row, col))

        # initialization algorithm dictionary
        init_funcs = {
            'gaussian': gaussian,
            'xavier': xavier,
            'zeros': zeros,
            'ones': ones
        }

        # return initialization algorithm
        return init_funcs[self._algorithm]

    def initialize(self, rows: int, columns: int):
        """
        'initialize' is a built-in function in the 'Initializers' class.
        This function initializes a numpy array based on the rows and columns.

        Arguments:
            rows: An integer of the rows in the numpy array.
            columns: An integer of the columns in the numpy array.

        Returns:
            A numpy array of initialized values.
        """
        if not isinstance(rows, int):
            # invalid datatype
            raise TypeError(f"'rows' is not an integer: {rows}")
        if not isinstance(columns, int):
            # invalid datatype
            raise TypeError(f"'columns' is not an integer: {columns}")

        # return numpy array
        return self._initializer(rows, columns)


class Activators:
    def __init__(self, algorithm: str, parameters: dict = None):
        """
        'Activators' is a class containing various activation algorithms.
        These activation algorithms currently include 'softmax' (Softmax), 'relu' (ReLU), 'lrelu' (Leaky ReLU), 'sigmoid' (Sigmoid), 'tanh' (Tanh), 'softplus' (Softplus), and 'mish' (Mish).
        The class structure promotes flexibility through the easy addition of other activation algorithms.

        Arguments:
            algorithm: A string referencing the desired activation algorithm.
            parameters: A dictionary referencing the parameters for the activation algorithm.
                (any parameters not referenced will be automatically defined)
        """
        # activation algorithms
        self.algorithms = [
            'softmax',
            'relu',
            'lrelu',
            'sigmoid',
            'tanh',
            'softplus',
            'mish'
        ]

        # internal activation algorithm parameters
        self._algorithm = self._check_algorithm(algorithm)
        self._params = self._get_params(parameters)

        # activation algorithms
        self._activator = self._get_activator()
        self._d_activator = self._get_d_activator()

    def _check_algorithm(self, algorithm):
        # checks whether the activation algorithm is valid
        if algorithm in self.algorithms:
            # valid activation algorithm
            return algorithm
        else:
            # invalid activation algorithm
            raise ValueError(
                f"Invalid activation algorithm: '{algorithm}'\n"
                f"Choose from: {[alg for alg in self.algorithms]}"
            )

    def _get_params(self, params):
        # defines activation algorithm parameters
        # default activation algorithm parameters
        default = {
            'softmax': None,
            'relu': None,
            'lrelu': {
                'beta': 0.01
            },
            'sigmoid': None,
            'tanh': None,
            'softplus': {
                'beta': 1
            },
            'mish': {
                'beta': 1
            }
        }

        # instantiate parameter dictionary
        prms = default[self._algorithm]

        if params and prms and isinstance(params, dict):
            # set defined parameter
            for prm in params:
                if prm in prms:
                    # todo: add errors for each parameter
                    # valid parameter
                    prms[prm] = params[prm]
                else:
                    # invalid parameter
                    raise UserWarning(
                        f"Invalid parameter for '{self._algorithm}': {prm}\n"
                        f"Choose from: {[prm for prm in prms]}"
                    )
        elif params and isinstance(params, dict):
            # parameters not taken
            raise UserWarning(f"'{self._algorithm}' does not take parameters")
        elif params:
            # invalid data type
            raise TypeError(
                f"'parameters' is not a dictionary: {params}\n"
                f"Choose from: {[prm for prm in prms]}"
            )

        # return parameters
        return prms

    def _get_activator(self):
        # defines activation algorithm
        # todo: check order of operations
        def softmax(x):
            # Softmax activation
            return np.exp(x) / np.sum(np.exp(x))

        def relu(x):
            # ReLU activation
            return np.maximum(0, x)

        def lrelu(x):
            # Leaky ReLU activation
            return np.maximum(self._params['beta'] * x, x)

        def sigmoid(x):
            # Sigmoid activation
            return 1 / (1 + np.exp(-x))

        def tanh(x):
            # Tanh activation
            return np.tanh(x)

        def softplus(x):
            # Softplus activation
            return np.log(1 + np.exp(self._params['beta'] * x)) / self._params['beta']

        def mish(x):
            # Mish activation
            return x * np.tanh(np.log(1 + np.exp(self._params['beta'] * x)) / self._params['beta'])

        # activation algorithm dictionary
        act_funcs = {
            'softmax': softmax,
            'relu': relu,
            'lrelu': lrelu,
            'sigmoid': sigmoid,
            'tanh': tanh,
            'softplus': softplus,
            'mish': mish
        }

        # return activation algorithm
        return act_funcs[self._algorithm]

    def _get_d_activator(self):
        # defines derivative of activation algorithm
        # todo: check order of operations
        def d_softmax(x):
            # derivative of Softmax activation
            return ...  # todo: write this algorithm

        def d_relu(x):
            # derivative of ReLU activation
            return np.where(x > 0, 1, 0)

        def d_lrelu(x):
            # derivative of Leaky ReLU activation
            return np.where(x > 0, 1, self._params['beta'])

        def d_sigmoid(x):
            # derivative of Sigmoid activation
            return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

        def d_tanh(x):
            # derivative of Tanh activation
            return np.cosh(x) ** -2

        def d_softplus(x):
            # derivative of Softplus activation
            return self._params['beta'] * np.exp(self._params['beta'] * x) / (self._params['beta'] + self._params['beta'] * np.exp(self._params['beta'] * x))

        def d_mish(x):
            # derivative of Mish activation
            return (np.tanh(np.log(1 + np.exp(self._params['beta'] * x)) / self._params['beta'])) + (x * (np.cosh(np.log(1 + np.exp(self._params['beta'] * x)) / self._params['beta']) ** -2) * (self._params['beta'] * np.exp(self._params['beta'] * x) / (self._params['beta'] + self._params['beta'] * np.exp(self._params['beta'] * x))))

        # derivative of activation algorithm dictionary
        d_act_funcs = {
            'softmax': d_softmax,
            'relu': d_relu,
            'lrelu': d_lrelu,
            'sigmoid': d_sigmoid,
            'tanh': d_tanh,
            'softplus': d_softplus,
            'mish': d_mish
        }

        # return derivative of activation algorithm
        return d_act_funcs[self._algorithm]

    def activate(self, x: np.ndarray):
        """
        'activate' is a built-in function in the 'Activators' class.
        This function runs a numpy array through an activation algorithm.

        Arguments:
            x: A numpy array of input activations.

        Returns:
            An numpy array of activated inputs.
        """
        if not isinstance(x, np.ndarray):
            # invalid datatype
            raise TypeError(f"'x' is not a numpy array: {x}")

        # return numpy array
        return self._activator(x)

    def d_activate(self, x: np.ndarray):
        """
        'd_activate' is a built-in function in the 'Activators' class.
        This function runs a numpy array through the derivative of an activation algorithm.

        Arguments:
            x: A numpy array of input activations.

        Returns:
            An numpy array of the derivative of activated inputs.
        """
        if not isinstance(x, np.ndarray):
            # invalid datatype
            raise TypeError(f"'x' is not a numpy array: {x}")

        # return numpy array
        return self._d_activator(x)


class Losses:
    def __init__(self, algorithm: str, parameters: dict = None):
        """
        'Losses' is a class containing various loss algorithms.
        These activation algorithms currently include 'ssr' (SSR), 'srsr' (SRSR), and 'centropy' (Cross-Entropy).
        The class structure promotes flexibility through the easy addition of other loss algorithms.

        Arguments:
            algorithm: A string referencing the desired loss algorithm.
            parameters: A dictionary referencing the parameters for the loss algorithm.
                (any parameters not referenced will be automatically defined)
        """
        # loss algorithms
        self.algorithms = [
            'centropy',
            'ssr',
            'srsr'
        ]

        # internal loss algorithm parameters
        self._algorithm = self._check_algorithm(algorithm)
        self._params = self._get_params(parameters)

        # loss algorithms
        self._loss = self._get_loss()
        self._d_loss = self._get_d_loss()

    def _check_algorithm(self, algorithm):
        # checks whether the loss algorithm is valid
        if algorithm in self.algorithms:
            # valid loss algorithm
            return algorithm
        else:
            # invalid loss algorithm
            raise ValueError(
                f"Invalid loss algorithm: '{algorithm}'\n"
                f"Choose from: {[alg for alg in self.algorithms]}"
            )

    def _get_params(self, params):
        # defines loss algorithm parameters
        # default loss algorithm parameters
        default = {
            'centropy': None,
            'ssr': None,
            'srsr': None
        }

        # instantiate parameter dictionary
        prms = default[self._algorithm]

        if params and prms and isinstance(params, dict):
            # set defined parameter
            for prm in params:
                if prm in prms:
                    # todo: add errors for each parameter
                    # valid parameter
                    prms[prm] = params[prm]
                else:
                    # invalid parameter
                    raise UserWarning(
                        f"Invalid parameter for '{self._algorithm}': {prm}\n"
                        f"Choose from: {[prm for prm in prms]}"
                    )
        elif params and isinstance(params, dict):
            # parameters not taken
            raise UserWarning(f"'{self._algorithm}' does not take parameters")
        elif params:
            # invalid data type
            raise TypeError(
                f"'parameters' is not a dictionary: {params}\n"
                f"Choose from: {[prm for prm in prms]}"
            )

        # return parameters
        return prms

    def _get_loss(self):
        # defines loss algorithm
        # todo: check order of operations
        def ssr(y, yhat):
            # SSR loss
            return np.sum((y - yhat) ** 2)

        def srsr(y, yhat):
            # SRSR loss
            return np.sum(np.abs(y - yhat))

        def centropy(y, yhat):
            # Cross-Entropy loss
            return np.sum(yhat * np.log(y))

        # loss algorithm dictionary
        loss_funcs = {
            'centropy': centropy,
            'ssr': ssr,
            'srsr': srsr
        }

        # return loss algorithm
        return loss_funcs[self._algorithm]

    def _get_d_loss(self):
        # defines derivative of loss algorithm
        # todo: check order of operations
        def d_ssr(y, yhat):
            # derivative of SSR loss
            return -2 * (y - yhat)

        def d_srsr(y, yhat):
            # derivative of SRSR loss
            return ...  # todo: write this algorithm

        def d_centropy(y, yhat):
            # derivative of Cross-Entropy loss
            return ...  # todo: write this algorithm

        # derivative of loss algorithm dictionary
        d_loss_funcs = {
            'centropy': d_centropy,
            'ssr': d_ssr,
            'srsr': d_srsr
        }

        # return derivative of loss algorithm
        return d_loss_funcs[self._algorithm]

    def loss(self, y: np.ndarray, yhat: np.ndarray):
        """
        'loss' is a built-in function in the 'Loss' class.
        This function runs a numpy array through a loss algorithm.

        Arguments:
            y: A numpy array of predicted activations.
            yhat: A numpy array of expected activations

        Returns:
            A numpy float of the calculated loss.
        """
        if not isinstance(y, np.ndarray):
            # invalid datatype
            raise TypeError(f"'y' is not a numpy array: {y}")
        if not isinstance(yhat, np.ndarray):
            # invalid datatype
            raise TypeError(f"'yhat' is not a numpy array: {yhat}")

        # return loss
        return self._loss(y, yhat)

    def d_loss(self, y: np.ndarray, yhat: np.ndarray):
        """
        'd_loss' is a built-in function in the 'Loss' class.
        This function runs a numpy array through the derivative of a loss algorithm.

        Arguments:
            y: A numpy array of predicted activations.
            yhat: A numpy array of expected activations

        Returns:
            A numpy array of the derivative of calculated loss with respect to the predicted activations.
        """
        if not isinstance(y, np.ndarray):
            # invalid datatype
            raise TypeError(f"'y' is not a numpy array: {y}")
        if not isinstance(yhat, np.ndarray):
            # invalid datatype
            raise TypeError(f"'yhat' is not a numpy array: {yhat}")

        # return numpy array
        return self._d_loss(y, yhat)


class Optimizers:
    def __init__(self, algorithm: str, hyperparameters: dict = None):
        """
        'Optimizers' is a class containing various optimization algorithms.
        These optimization algorithms currently include 'adam' (Adam), 'sgd' (SGD), and 'rms' (RMSprop).
        The class structure promotes flexibility through the easy addition of other optimization algorithms.

        Arguments:
            algorithm: A string referencing the desired optimization algorithm.
            hyperparameters: A dictionary referencing the hyperparameters for the optimization algorithm.
                (any hyperparameters not referenced will be automatically defined)
        """
        # optimization algorithms
        self.algorithms = [
            'adam',
            'sgd',
            'rms'
        ]

        # internal optimization algorithm parameters
        self._algorithm = self._check_algorithm(algorithm)
        self._hyps = self._get_hyperparams(hyperparameters)

        # optimization algorithm
        self._optim = self._get_optim()

        # internal memory
        self._memory = self._get_memory()

    def _check_algorithm(self, algorithm):
        # checks whether the optimization algorithm is valid
        if algorithm in self.algorithms:
            # valid optimization algorithm
            return algorithm
        else:
            # invalid optimization algorithm
            raise ValueError(
                f"Invalid optimization algorithm: '{algorithm}'\n"
                f"Choose from: {[alg for alg in self.algorithms]}"
            )

    def _get_hyperparams(self, hyperparams):
        # defines optimization algorithm hyperparameters
        # default optimization algorithm hyperparameters
        default = {
            'adam': {
                'gamma': 0.001,
                'lambda_d': 0,
                'beta': (0.9, 0.999),
                'epsilon': 1e-8,
                'ams': False
            },
            'sgd': {
                'gamma': 0.001,
                'lambda_d': 0,
                'mu': 0,
                'tau': 0,
                'nesterov': False
            },
            'rms': {
                'gamma': 0.001,
                'lambda_d': 0,
                'beta': 0.99,
                'mu': 0,
                'epsilon': 1e-8
            }
        }

        # instantiate hyperparameter dictionary
        hyps = default[self._algorithm]

        if hyperparams and isinstance(hyperparams, dict):
            # set defined hyperparameters
            for hyp in hyperparams:
                if hyp in hyps:
                    # todo: add errors for each hyperparameter
                    # valid hyperparameter
                    hyps[hyp] = hyperparams[hyp]
                else:
                    # invalid hyperparameter
                    raise UserWarning(
                        f"Invalid hyperparameter for '{self._algorithm}': {hyp}\n"
                        f"Choose from: {[hyp for hyp in hyps]}"
                    )
        elif hyperparams:
            # invalid data type
            raise TypeError(
                f"'hyperparameters' is not a dictionary: {hyperparams}\n"
                f"Choose from: {[hyp for hyp in hyps]}"
            )

        # return hyperparameters
        return hyps

    def _get_optim(self):
        # defines optimization algorithm
        # todo: check order of operations
        def adam(thetas, nablas):
            # Adam optimization algorithm
            # weight decay
            deltas = nablas + (self._hyps['lambda_d'] * thetas)
            if self._memory['deltas_p']:
                # momentum
                deltas = ((self._hyps['beta'][0] * self._memory['deltas_p']) + ((1 - self._hyps['beta'][0]) * deltas)) / (1 - self._hyps['beta'][0])
            if self._memory['upsilons_p']:
                # square momentum
                upsilons = ((self._hyps['beta'][1] * self._memory['upsilons_p']) / (1 - self._hyps['beta'][1])) + (deltas ** 2)
            else:
                # square
                upsilons = deltas ** 2
            if self._hyps['ams']:
                # ams-grad variant
                if self._memory['upsilons_mx']:
                    upsilons_mx = np.maximum(self._memory['upsilons_mx'], upsilons)
                else:
                    upsilons_mx = upsilons
                # update memory
                self._memory['upsilons_mx'] = upsilons_mx
                # set upsilons
                upsilons = upsilons_mx
            # optimization
            thetas -= self._hyps['gamma'] * deltas / (np.sqrt(upsilons) + self._hyps['epsilon'])

            # update memory
            self._memory['deltas_p'] = deltas
            self._memory['upsilons_p'] = upsilons

            # return thetas
            return thetas

        def sgd(thetas, nablas):
            # SGD optimization algorithm
            # weight decay
            deltas = nablas + (self._hyps['lambda_d'] * thetas)
            if self._hyps['mu'] and self._memory['deltas_p']:
                # momentum
                deltas = (self._hyps['mu'] * self._memory['deltas_p']) + ((1 - self._hyps['tau']) * deltas)
            if self._hyps['nesterov'] and self._hyps['mu']:
                # nesterov momentum
                deltas += nablas + self._hyps['mu'] * deltas
            # optimization
            thetas -= self._hyps['gamma'] * deltas

            # update memory
            self._memory['deltas_p'] = deltas

            # return thetas
            return thetas

        def rms(thetas, nablas):
            # RMSprop optimization algorithm
            # weight decay
            deltas = nablas + (self._hyps['lambda_d'] * thetas)
            if self._memory['upsilons_p']:
                # square momentum
                upsilons = (self._hyps['beta'] * self._memory['upsilons_p']) + ((1 - self._hyps['beta']) * (deltas ** 2))
            else:
                # square
                upsilons = (1 - self._hyps['beta']) * (deltas ** 2)
            # step calculation
            deltas = deltas / (np.sqrt(upsilons) + self._hyps['epsilon'])
            if self._hyps['mu'] and self._memory['deltas_p']:
                # momentum
                deltas += self._hyps['mu'] * self._memory['deltas_p']
            # optimization
            thetas -= self._hyps['gamma'] * deltas

            # update memory
            self._memory['deltas_p'] = deltas
            self._memory['upsilons_p'] = upsilons

            # return thetas
            return thetas

        # optimization algorithm dictionary
        optim_funcs = {
            'adam': adam,
            'sgd': sgd,
            'rms': rms
        }

        # return optimization algorithm
        return optim_funcs[self._algorithm]

    def _get_memory(self):
        # instantiates memory dictionary
        # required memory components for each optimization algorithm
        memories = {
            'adam': {
                'deltas_p': None,
                'upsilons_p': None,
                'upsilons_mx': None
            },
            'sgd': {
                'deltas_p': None
            },
            'rms': {
                'deltas_p': None,
                'upsilons_p': None
            }
        }
        # return memory dictionary
        return memories[self._algorithm]

    def optimize(self, thetas: np.ndarray, nablas: np.ndarray):
        """
        'update' is a built-in function in the 'Optimizers' class.
        This function updates the parameters of a model based on the gradients.

        Arguments:
            thetas: A numpy array of the parameters that will be optimized.
            nablas: A numpy array of the gradients used to optimize the parameters.

        Returns:
            A numpy array of updated parameters.
        """
        if not isinstance(thetas, np.ndarray):
            # invalid datatype
            raise TypeError(f"'thetas' is not a numpy array: {thetas}")
        if not isinstance(nablas, np.ndarray):
            # invalid datatype
            raise TypeError(f"'nablas' is not a numpy array: {nablas}")

        # return updated thetas
        return self._optim(thetas, nablas)
