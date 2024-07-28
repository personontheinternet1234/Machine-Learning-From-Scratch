r"""
'algorithms' includes mathematical algorithms for GardenPy.

'algorithms' includes:
    'Initializers': Initialization algorithms for kernels/weights/biases.
    'Activators': Activation algorithms for activations.
    'Losses': Loss algorithms for loss.
    'Optimizers': Optimization algorithms for kernels/weights/biases.

Refer to 'todo' for in-depth documentation on these algorithms.
"""

from typing import Union
import warnings

import numpy as np

from .objects import Tensor


class Initializers:
    def __init__(self, algorithm: str, parameters: dict = None, **kwargs):
        r"""
        'Initializers' is a class containing various initialization algorithms.
        These initialization algorithms currently include 'xavier' (Xavier/Glorot), 'gaussian' (Gaussian/Normal), 'zeros' (Uniform Zeros), and 'ones' (Uniform Ones).
        The class structure promotes flexibility through the easy addition of other initialization algorithms.

        Arguments:
            algorithm: A string referencing the desired initialization algorithm.
            parameters: A dictionary referencing the parameters for the initialization algorithm.
            (any parameters not referenced will be automatically defined)
                'xavier' (Xavier initialization):
                    'gain': The gain value.
            kwargs: Key-word arguments that can be used instead of a dictionary for the parameters.
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
        self._params = self._get_params(parameters, kwargs)

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

    def _get_params(self, params, kwargs):
        # defines initialization algorithm parameters
        # default initialization algorithm parameters
        default = {
            'xavier': {
                'gain': 1.0
            },
            'gaussian': None,
            'zeros': None,
            'ones': None
        }
        # default initialization algorithm parameter datatypes
        dtypes = {
            'xavier': {
                'gain': (float, int)
            }
        }
        # default initialization algorithm parameter value types
        vtypes = {
            'xavier': {
                'gain': lambda x: 0.0 < x
            }
        }
        # default optimization algorithm hyperparameter conversion types
        ctypes = {
            'xavier': {
                'gain': lambda x: float(x)
            }
        }

        # instantiate parameter dictionary
        prms = default[self._algorithm]

        # combine keyword arguments and parameters
        if params and kwargs:
            params.update(kwargs)
        elif kwargs:
            params = kwargs

        if params and (prms is not None) and isinstance(params, dict):
            # set defined parameter
            for prm in params:
                if prm not in prms:
                    # invalid parameter
                    print()
                    warnings.warn(
                        f"\nInvalid parameter for '{self._algorithm}': '{prm}'\n"
                        f"Choose from: '{[prm for prm in prms]}'",
                        UserWarning
                    )
                elif prm in prms and not isinstance(params[prm], dtypes[self._algorithm][prm]):
                    # invalid datatype for parameter
                    raise TypeError(
                        f"Invalid datatype for '{self._algorithm}': '{prm}'\n"
                        f"Choose from: {dtypes[self._algorithm][prm]}"
                    )
                elif prm in prms and not (vtypes[self._algorithm][prm](params[prm])):
                    # invalid value for parameter
                    raise TypeError(
                        f"Invalid value for '{self._algorithm}': '{prm}'\n"
                        f"Conditional: {vtypes[self._algorithm][prm]}"
                    )
                else:
                    # valid parameter
                    prms[prm] = ctypes[self._algorithm][prm](params[prm])
        elif params and isinstance(params, dict):
            # parameters not taken
            print()
            warnings.warn(f"\n'{self._algorithm}' does not take parameters", UserWarning)
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
        def xavier(row, col):
            # Xavier/Glorot initialization
            return np.random.randn(row, col) * self._params['gain'] * np.sqrt(2.0 / float(row + col))

        def gaussian(row, col):
            # Gaussian initialization
            return np.random.randn(row, col)

        def zeros(row, col):
            # Zeros uniform initialization
            return np.zeros((row, col), dtype=np.float64)

        def ones(row, col):
            # Ones uniform initialization
            return np.ones((row, col), dtype=np.float64)

        # initialization algorithm dictionary
        init_funcs = {
            'xavier': xavier,
            'gaussian': gaussian,
            'zeros': zeros,
            'ones': ones
        }

        # return initialization algorithm
        return init_funcs[self._algorithm]

    def initialize(self, rows: int, columns: int, tens=True) -> Union[Tensor, np.ndarray]:
        r"""
        'initialize' is a built-in function in the 'Initializers' class.
        This function initializes a Tensor based on the rows and columns.

        Arguments:
            rows: An integer of the rows in the Tensor.
            columns: An integer of the columns in the Tensor.
            tens: Bool to return a tensor or NumPy Array.

        Returns:
            A Tensor of initialized values.
        """
        if not isinstance(rows, int):
            # invalid datatype
            raise TypeError(f"'rows' is not an integer: '{rows}'")
        if not isinstance(columns, int):
            # invalid datatype
            raise TypeError(f"'columns' is not an integer: '{columns}'")

        # return initialized tensor
        if tens:
            return Tensor(self._initializer(rows, columns))
        else:
            return self._initializer(rows, columns)


class Activators:
    def __init__(self, algorithm: str, parameters: dict = None, **kwargs):
        r"""
        'Activators' is a class containing various activation algorithms.
        These activation algorithms currently include 'softmax' (Softmax), 'relu' (ReLU), 'lrelu' (Leaky ReLU), 'sigmoid' (Sigmoid), 'tanh' (Tanh), 'softplus' (Softplus), and 'mish' (Mish).
        The class structure promotes flexibility through the easy addition of other activation algorithms.

        Arguments:
            algorithm: A string referencing the desired activation algorithm.
            parameters: A dictionary referencing the parameters for the activation algorithm.
            (any parameters not referenced will be automatically defined)
                'lrelu' (Leaky ReLU):
                    'beta': The negative slope coefficient.
                'softplus' (Softplus):
                    'beta': The Beta value.
                'mish' (Mish):
                    'beta': The Beta value.
            kwargs: Key-word arguments that can be used instead of a dictionary for the parameters.
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
        self._params = self._get_params(parameters, kwargs)

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
                f"Choose from: '{[alg for alg in self.algorithms]}'"
            )

    def _get_params(self, params, kwargs):
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
                'beta': 1.0
            },
            'mish': {
                'beta': 1.0
            }
        }
        # default activation algorithm parameter datatypes
        dtypes = {
            'lrelu': {
                'beta': (float, int)
            },
            'softplus': {
                'beta': (float, int)
            },
            'mish': {
                'beta': (float, int)
            }
        }
        # default activation algorithm parameter value types
        vtypes = {
            'lrelu': {
                'beta': lambda x: 0.0 < x
            },
            'softplus': {
                'beta': lambda x: 0.0 <= x
            },
            'mish': {
                'beta': lambda x: 0.0 <= x
            }
        }
        # default optimization algorithm hyperparameter conversion types
        ctypes = {
            'lrelu': {
                'beta': lambda x: float(x)
            },
            'softplus': {
                'beta': lambda x: float(x)
            },
            'mish': {
                'beta': lambda x: float(x)
            }
        }

        # instantiate parameter dictionary
        prms = default[self._algorithm]

        # combine keyword arguments and parameters
        if params and kwargs:
            params.update(kwargs)
        elif kwargs:
            params = kwargs

        if params and (prms is not None) and isinstance(params, dict):
            # set defined parameter
            for prm in params:
                if prm not in prms:
                    # invalid parameter
                    print()
                    warnings.warn(
                        f"\nInvalid parameter for '{self._algorithm}': '{prm}'\n"
                        f"Choose from: '{[prm for prm in prms]}'",
                        UserWarning
                    )
                elif prm in prms and not isinstance(params[prm], dtypes[self._algorithm][prm]):
                    # invalid datatype for parameter
                    raise TypeError(
                        f"Invalid datatype for '{self._algorithm}': '{prm}'\n"
                        f"Choose from: '{dtypes[self._algorithm][prm]}'"
                    )
                elif prm in prms and not (vtypes[self._algorithm][prm](params[prm])):
                    # invalid value for parameter
                    raise TypeError(
                        f"Invalid value for '{self._algorithm}': '{prm}'\n"
                        f"Conditional: '{vtypes[self._algorithm][prm]}'"
                    )
                else:
                    # valid parameter
                    prms[prm] = ctypes[self._algorithm][prm](params[prm])
        elif params and isinstance(params, dict):
            # parameters not taken
            print()
            warnings.warn(f"\n'{self._algorithm}' does not take parameters", UserWarning)
        elif params:
            # invalid data type
            raise TypeError(
                f"'parameters' is not a dictionary: '{params}'\n"
                f"Choose from: '{[prm for prm in prms]}'"
            )

        # return parameters
        return prms

    def _get_activator(self):
        # defines activation algorithm
        def softmax(x):
            # Softmax activation
            return np.exp(x) / np.sum(np.exp(x))

        def relu(x):
            # ReLU activation
            return np.maximum(0.0, x)

        def lrelu(x):
            # Leaky ReLU activation
            return np.maximum(self._params['beta'] * x, x)

        def sigmoid(x):
            # Sigmoid activation
            return (1.0 + np.exp(-x)) ** -1.0

        def tanh(x):
            # Tanh activation
            return np.tanh(x)

        def softplus(x):
            # Softplus activation
            return np.log(1.0 + np.exp(self._params['beta'] * x)) / self._params['beta']

        def mish(x):
            # Mish activation
            return x * np.tanh(np.log(1.0 + np.exp(self._params['beta'] * x)) / self._params['beta'])

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
        def d_softmax(x, _=None):
            # derivative of Softmax activation
            return (np.exp(x) * np.sum(np.exp(x)) - np.exp(2 * x)) / (np.sum(np.exp(x)) ** 2)

        def d_relu(x, _=None):
            # derivative of ReLU activation
            return np.where(x > 0.0, 1.0, 0.0)

        def d_lrelu(x, _=None):
            # derivative of Leaky ReLU activation
            return np.where(x > 0.0, 1.0, self._params['beta'])

        def d_sigmoid(x, _=None):
            # derivative of Sigmoid activation
            return np.exp(-x) / ((1.0 + np.exp(-x)) ** 2.0)

        def d_tanh(x, _=None):
            # derivative of Tanh activation
            return np.cosh(x) ** -2.0

        def d_softplus(x, _=None):
            # derivative of Softplus activation
            return self._params['beta'] * np.exp(self._params['beta'] * x) / (self._params['beta'] + self._params['beta'] * np.exp(self._params['beta'] * x))

        def d_mish(x, _=None):
            # derivative of Mish activation
            # todo: check order of operations
            return (np.tanh(np.log(1.0 + np.exp(self._params['beta'] * x)) / self._params['beta'])) + (x * (np.cosh(np.log(1.0 + np.exp(self._params['beta'] * x)) / self._params['beta']) ** -2.0) * (self._params['beta'] * np.exp(self._params['beta'] * x) / (self._params['beta'] + self._params['beta'] * np.exp(self._params['beta'] * x))))

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

    def activate(self, x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        r"""
        'activate' is a built-in function in the 'Activators' class.
        This function runs a Tensor through an activation algorithm.
        This function contains built-in compatibility with automatic differentiation for Tensors.

        Arguments:
            x: A Tensor or NumPy Array of input activations.

        Returns:
            An Tensor or NumPy Array of activated inputs.
        """
        if not isinstance(x, (Tensor, np.ndarray)):
            # invalid datatype
            raise TypeError(f"'x' is not a Tensor or NumPy Array: '{x}'")

        if isinstance(x, Tensor):
            # calculate result
            result = Tensor(self._activator(x.to_array()))
            # track operation
            x.tracker['opr'].append('activate')
            x.tracker['drv'].append(self._d_activator)
            # track origin
            result.tracker['org'] = [x, '_']
            # track relation
            x.tracker['rlt'].append(['_', result])

            # return result
            return result
        else:
            # return result
            return self._activator(x)

    def d_activate(self, x: np.ndarray) -> np.ndarray:
        r"""
        'd_activate' is a built-in function in the 'Activators' class.
        This function runs a NumPy Array through the derivative of an activation algorithm.
        This function exists for manual differentiation, use 'nabla' for Tensor support.

        Arguments:
            x: A NumPy Array of input activations.

        Returns:
            An NumPy Array of the derivative of activated inputs.
        """
        if not isinstance(x, np.ndarray):
            # invalid datatype
            raise TypeError(f"'x' is not a NumPy Array: '{x}'")

        # return numpy array
        return self._d_activator(x)


class Losses:
    def __init__(self, algorithm: str, parameters: dict = None, **kwargs):
        r"""
        'Losses' is a class containing various loss algorithms.
        These activation algorithms currently include 'ssr' (Sum of the Squared Residuals), 'savr' (Sum of the Absolute Valued Residuals), and 'centropy' (Cross-Entropy).
        The class structure promotes flexibility through the easy addition of other loss algorithms.

        Arguments:
            algorithm: A string referencing the desired loss algorithm.
            parameters: A dictionary referencing the parameters for the loss algorithm.
            (any parameters not referenced will be automatically defined)
                Currently, 'Losses' takes no arguments.
            kwargs: Key-word arguments that can be used instead of a dictionary for the parameters.
        """
        # loss algorithms
        self.algorithms = [
            'centropy',
            'ssr',
            'savr'
        ]

        # internal loss algorithm parameters
        self._algorithm = self._check_algorithm(algorithm)
        self._params = self._get_params(parameters, kwargs)

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
                f"Choose from: '{[alg for alg in self.algorithms]}'"
            )

    def _get_params(self, params, kwargs):
        # defines loss algorithm parameters
        # default loss algorithm parameters
        default = {
            'centropy': None,
            'ssr': None,
            'savr': None
        }
        # default loss algorithm parameter datatypes
        dtypes = {
        }
        # default loss algorithm parameter value types
        vtypes = {
        }
        # default optimization algorithm hyperparameter conversion types
        ctypes = {
        }

        # instantiate parameter dictionary
        prms = default[self._algorithm]

        # combine keyword arguments and parameters
        if params and kwargs:
            params.update(kwargs)
        elif kwargs:
            params = kwargs

        if params and (prms is not None) and isinstance(params, dict):
            # set defined parameter
            for prm in params:
                if prm not in prms:
                    # invalid parameter
                    print()
                    warnings.warn(
                        f"\nInvalid parameter for '{self._algorithm}': '{prm}'\n"
                        f"Choose from: '{[prm for prm in prms]}'",
                        UserWarning
                    )
                elif prm in prms and not isinstance(params[prm], dtypes[self._algorithm][prm]):
                    # invalid datatype for parameter
                    raise TypeError(
                        f"Invalid datatype for '{self._algorithm}': '{prm}'\n"
                        f"Choose from: '{dtypes[self._algorithm][prm]}'"
                    )
                elif prm in prms and not (vtypes[self._algorithm][prm](params[prm])):
                    # invalid value for parameter
                    raise TypeError(
                        f"Invalid value for '{self._algorithm}': '{prm}'\n"
                        f"Conditional: '{vtypes[self._algorithm][prm]}'"
                    )
                else:
                    # valid parameter
                    prms[prm] = ctypes[self._algorithm][prm](params[prm])
        elif params and isinstance(params, dict):
            # parameters not taken
            print()
            warnings.warn(f"\n'{self._algorithm}' does not take parameters", UserWarning)
        elif params:
            # invalid data type
            raise TypeError(
                f"'parameters' is not a dictionary: '{params}'\n"
                f"Choose from: '{[prm for prm in prms]}'"
            )

        # return parameters
        return prms

    def _get_loss(self):
        # defines loss algorithm
        def centropy(yhat, y):
            # Cross-Entropy loss
            return -np.sum(y * np.log(yhat))

        def ssr(yhat, y):
            # Sum of the Squared Residuals loss
            return np.sum((y - yhat) ** 2.0)

        def savr(yhat, y):
            # Sum of the Absolute Valued Residuals loss
            return np.sum(np.abs(y - yhat))

        # loss algorithm dictionary
        loss_funcs = {
            'centropy': centropy,
            'ssr': ssr,
            'savr': savr
        }

        # return loss algorithm
        return loss_funcs[self._algorithm]

    def _get_d_loss(self):
        # defines derivative of loss algorithm
        def d_centropy(yhat, y):
            # derivative of Cross-Entropy loss
            return -np.log(yhat) - (y / yhat)

        def d_ssr(yhat, y):
            # derivative of Sum of the Squared Residuals loss
            return -2.0 * (y - yhat)

        def d_savr(yhat, y):
            # derivative of Sum of the Absolute Valued Residuals loss
            return -np.sign(y - yhat)

        # derivative of loss algorithm dictionary
        d_loss_funcs = {
            'centropy': d_centropy,
            'ssr': d_ssr,
            'savr': d_savr
        }

        # return derivative of loss algorithm
        return d_loss_funcs[self._algorithm]

    def loss(self, yhat: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        r"""
        'loss' is a built-in function in the 'Loss' class.
        This function runs a NumPy Array through a loss algorithm.
        This function contains built-in compatibility with automatic differentiation for Tensors.

        Arguments:
            yhat: A Tensor or NumPy Array of predicted activations.
            y: A Tensor or NumPy Array of expected activations.

        Returns:
            A Tensor or NumPy Array of the calculated loss.
        """
        if not isinstance(yhat, (Tensor, np.ndarray)):
            # invalid datatype
            raise TypeError(f"'yhat' is not a Tensor or NumPy Array: '{yhat}'")
        if not isinstance(y, (Tensor, np.ndarray)):
            # invalid datatype
            raise TypeError(f"'y' is not a Tensor or NumPy Array: '{y}'")
        y_arr = y
        if isinstance(y, Tensor):
            # convert to numpy array to avoid unnecessary tracking
            y_arr = y.to_array()

        if isinstance(yhat, Tensor):
            # calculate result
            result = Tensor([self._loss(yhat.to_array(), y_arr)])
            # track operation
            yhat.tracker['opr'].append('loss')
            yhat.tracker['drv'].append(self._d_loss)
            # track origin
            result.tracker['org'] = [yhat, y]
            # track relation
            yhat.tracker['rlt'].append([y, result])

            # return result
            return result
        else:
            # return result
            return self._loss(yhat, y_arr)

    def d_loss(self, yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""
        'd_loss' is a built-in function in the 'Loss' class.
        This function runs a NumPy Array through the derivative of a loss algorithm.
        This function exists for manual differentiation; use 'nabla' for Tensor support.

        Arguments:
            yhat: A NumPy Array of expected activations.
            y: A NumPy Array of predicted activations.

        Returns:
            A NumPy Array of the derivative of calculated loss with respect to the predicted activations.
        """
        if not isinstance(yhat, np.ndarray):
            # invalid datatype
            raise TypeError(f"'yhat' is not a NumPy Array: '{yhat}'")
        if not isinstance(y, np.ndarray):
            # invalid datatype
            raise TypeError(f"'y' is not a NumPy Array: '{y}'")

        # return numpy array
        return self._d_loss(yhat, y)


class Optimizers:
    def __init__(self, algorithm: str, hyperparameters: dict = None, **kwargs):
        r"""
        'Optimizers' is a class containing various optimization algorithms.
        These optimization algorithms currently include 'adam' (Adaptive Moment Estimation), 'sgd' (Stochastic Gradient Descent), and 'rmsp' (Root Mean Squared Propagation).
        The class structure promotes flexibility through the easy addition of other optimization algorithms.

        Arguments:
            algorithm: A string referencing the desired optimization algorithm.
            hyperparameters: A dictionary referencing the hyperparameters for the optimization algorithm.
            (any hyperparameters not referenced will be automatically defined)
                'adam' (Adaptive Moment Estimation):
                    'gamma': The learning rate value.
                    'lambda_d': The weight decay (L2 regularization) coefficient.
                    'beta1': The value for the weight held by the new delta.
                    'beta2': The value for the weight held by the new upsilon.
                    'epsilon': The numerical stability value to prevent division by zero.
                    'ams': A bool for the AMSGrad variant.
                'sgd' (Stochastic Gradient Descent):
                    'gamma': The learning rate value.
                    'lambda_d': The weight decay (L2 regularization) coefficient.
                    'mu': The value for the weight held by the previous delta.
                    'tau': the value for the dampening of the new delta.
                    'nesterov': A bool for Nesterov momentum.
                'rmsp' (Root Mean Squared Propagation):
                    'gamma': The learning rate value.
                    'lambda_d': The weight decay (L2 regularization) coefficient.
                    'beta': The value for the weight held by the new upsilon.
                    'mu': The value for the weight held by the previous delta.
                    'epsilon': The numerical stability value to prevent division by zero.
            kwargs: Key-word arguments that can be used instead of a dictionary for the hyperparameters.
        """
        # optimization algorithms
        self.algorithms = [
            'adam',
            'sgd',
            'rmsp'
        ]

        # internal optimization algorithm parameters
        self._algorithm = self._check_algorithm(algorithm)
        self._hyps = self._get_hyperparams(hyperparameters, kwargs)

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
                f"Choose from: '{[alg for alg in self.algorithms]}'"
            )

    def _get_hyperparams(self, hyperparams, kwargs):
        # defines optimization algorithm hyperparameters
        # default optimization algorithm hyperparameters
        default = {
            'adam': {
                'gamma': 1e-3,
                'lambda_d': 0.0,
                'beta1': 0.9,
                'beta2': 0.999,
                'epsilon': 1e-10,
                'ams': False
            },
            'sgd': {
                'gamma': 1e-3,
                'lambda_d': 0.0,
                'mu': 0.0,
                'tau': 0.0,
                'nesterov': False
            },
            'rmsp': {
                'gamma': 1e-3,
                'lambda_d': 0.0,
                'beta': 0.99,
                'mu': 0.0,
                'epsilon': 1e-10
            }
        }
        # default optimization algorithm hyperparameter datatypes
        dtypes = {
            'adam': {
                'gamma': (float, int),
                'lambda_d': (float, int),
                'beta1': (float, int),
                'beta2': (float, int),
                'epsilon': (float, int),
                'ams': (bool, int)
            },
            'sgd': {
                'gamma': (float, int),
                'lambda_d': (float, int),
                'mu': (float, int),
                'tau': (float, int),
                'nesterov': (bool, int)
            },
            'rmsp': {
                'gamma': (float, int),
                'lambda_d': (float, int),
                'beta': (float, int),
                'mu': (float, int),
                'epsilon': (float, int)
            }
        }
        # default optimization algorithm hyperparameter value types
        vtypes = {
            'adam': {
                'gamma': lambda x: True,
                'lambda_d': lambda x: 0.0 <= x,
                'beta1': lambda x: 0.0 < x,
                'beta2': lambda x: 0.0 < x,
                'epsilon': lambda x: 0.0 < x,
                'ams': lambda x: True
            },
            'sgd': {
                'gamma': lambda x: True,
                'lambda_d': lambda x: 0.0 <= x,
                'mu': lambda x: 0.0 <= x,
                'tau': lambda x: 0.0 <= x,
                'nesterov': lambda x: True
            },
            'rmsp': {
                'gamma': lambda x: True,
                'lambda_d': lambda x: 0.0 <= x,
                'beta': lambda x: 0.0 < x,
                'mu': lambda x: 0.0 < x,
                'epsilon': lambda x: 0.0 < x
            }
        }
        # default optimization algorithm hyperparameter conversion types
        ctypes = {
            'adam': {
                'gamma': lambda x: float(x),
                'lambda_d': lambda x: float(x),
                'beta1': lambda x: float(x),
                'beta2': lambda x: float(x),
                'epsilon': lambda x: float(x),
                'ams': lambda x: bool(x)
            },
            'sgd': {
                'gamma': lambda x: float(x),
                'lambda_d': lambda x: float(x),
                'mu': lambda x: float(x),
                'tau': lambda x: float(x),
                'nesterov': lambda x: bool(x)
            },
            'rmsp': {
                'gamma': lambda x: float(x),
                'lambda_d': lambda x: float(x),
                'beta': lambda x: float(x),
                'mu': lambda x: float(x),
                'epsilon': lambda x: float(x)
            }
        }

        # instantiate hyperparameters dictionary
        hyps = default[self._algorithm]

        # combine keyword arguments and hyperparameters
        if hyperparams and kwargs:
            hyperparams.update(kwargs)
        elif kwargs:
            hyperparams = kwargs

        if hyperparams and (hyps is not None) and isinstance(hyps, dict):
            # set defined hyperparameters
            for hyp in hyperparams:
                if hyp not in hyps:
                    # invalid hyperparameter
                    print()
                    warnings.warn(
                        f"\nInvalid hyperparameter for '{self._algorithm}': '{hyp}'\n"
                        f"Choose from: '{[hyp for hyp in hyps]}'",
                        UserWarning
                    )
                elif hyp in hyps and not isinstance(hyperparams[hyp], dtypes[self._algorithm][hyp]):
                    # invalid datatype for hyperparameter
                    raise TypeError(
                        f"Invalid datatype for '{self._algorithm}': '{hyp}'\n"
                        f"Choose from: '{dtypes[self._algorithm][hyp]}'"
                    )
                elif hyp in hyps and not (vtypes[self._algorithm][hyp](hyperparams[hyp])):
                    # invalid value for hyperparameter
                    raise TypeError(
                        f"Invalid value for '{self._algorithm}': '{hyp}'\n"
                        f"Conditional: '{vtypes[self._algorithm][hyp]}'"
                    )
                else:
                    # valid hyperparameter
                    hyps[hyp] = ctypes[self._algorithm][hyp](hyperparams[hyp])
        elif hyperparams and isinstance(hyperparams, dict):
            # hyperparameters not taken
            print()
            warnings.warn(f"\n'{self._algorithm}' does not take hyperparameters", UserWarning)
        elif hyperparams:
            # invalid data type
            raise TypeError(
                f"'hyperparameters' is not a dictionary: '{hyperparams}'\n"
                f"Choose from: '{[hyp for hyp in hyps]}'"
            )

        # return hyperparameters
        return hyps

    def _get_memory(self):
        # instantiates memory dictionary
        # required memory components for each optimization algorithm
        memories = {
            'adam': {
                'deltas_p': None,
                'upsilons_p': None,
                'upsilons_hat_mx': None
            },
            'sgd': {
                'deltas_p': None
            },
            'rmsp': {
                'deltas_p': None,
                'upsilons_p': None
            }
        }
        # return memory dictionary
        return memories[self._algorithm]

    def _get_optim(self):
        # defines optimization algorithm
        # todo: check order of operations
        def adam(thetas, nablas):
            # Adaptive Moment Estimation optimization algorithm
            # weight decay
            deltas = nablas + (self._hyps['lambda_d'] * thetas)
            if self._memory['deltas_p'] is not None:
                # momentum
                deltas = ((self._hyps['beta1'] * self._memory['deltas_p']) + ((1.0 - self._hyps['beta1']) * deltas))
            if self._memory['upsilons_p'] is not None:
                # square momentum
                upsilons = (self._hyps['beta2'] * self._memory['upsilons_p']) + (deltas ** 2.0)
            else:
                # square
                upsilons = deltas ** 2.0
            # hat
            deltas_hat = deltas / (1 - self._hyps['beta1'])
            upsilons_hat = upsilons / (1 - self._hyps['beta2'])
            if self._hyps['ams']:
                # ams-grad variant
                if self._memory['upsilons_hat_mx'] is not None:
                    upsilons_hat_mx = np.maximum(self._memory['upsilons_hat_mx'], upsilons_hat)
                else:
                    upsilons_hat_mx = upsilons_hat
                # update memory
                self._memory['upsilons_hat_mx'] = upsilons_hat_mx
                # set upsilons
                upsilons_hat = upsilons_hat_mx
            # optimization
            thetas -= self._hyps['gamma'] * deltas_hat / (np.sqrt(upsilons_hat) + self._hyps['epsilon'])

            # update memory
            self._memory['deltas_p'] = deltas
            self._memory['upsilons_p'] = upsilons

            # return thetas
            return thetas

        def sgd(thetas, nablas):
            # Stochastic Gradient Descent optimization algorithm
            # weight decay
            deltas = nablas + (self._hyps['lambda_d'] * thetas)
            if self._hyps['mu'] and (self._memory['deltas_p'] is not None):
                # momentum
                deltas = (self._hyps['mu'] * self._memory['deltas_p']) + ((1.0 - self._hyps['tau']) * deltas)
            if self._hyps['nesterov'] and (self._hyps['mu'] is not None):
                # nesterov momentum
                deltas += nablas + self._hyps['mu'] * deltas
            # optimization
            thetas -= self._hyps['gamma'] * deltas

            # update memory
            self._memory['deltas_p'] = deltas

            # return thetas
            return thetas

        def rmsp(thetas, nablas):
            # Root Mean Squared Propagation optimization algorithm
            # weight decay
            deltas = nablas + (self._hyps['lambda_d'] * thetas)
            if self._memory['upsilons_p'] is not None:
                # square momentum
                upsilons = (self._hyps['beta'] * self._memory['upsilons_p']) + ((1.0 - self._hyps['beta']) * (deltas ** 2.0))
            else:
                # square
                upsilons = (1.0 - self._hyps['beta']) * (deltas ** 2.0)
            # step calculation
            deltas = deltas / (np.sqrt(upsilons) + self._hyps['epsilon'])
            if self._hyps['mu'] and (self._memory['deltas_p'] is not None):
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
            'rmsp': rmsp
        }

        # return optimization algorithm
        return optim_funcs[self._algorithm]

    def optimize(self, thetas: Union[Tensor, np.ndarray], nablas: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        r"""
        'optimize' is a built-in function in the 'Optimizers' class.
        This function updates the parameters of a model based on the gradients.
        This function contains built-in compatibility with automatic differentiation for Tensors.

        Arguments:
            thetas: A Tensor or NumPy Array of the parameters that will be optimized.
            nablas: A Tensor or NumPy Array of the gradients used to optimize the parameters.

        Returns:
            A Tensor or NumPy Array of updated parameters.
        """
        if not isinstance(thetas, (Tensor, np.ndarray)):
            # invalid datatype
            raise TypeError(f"'thetas' is not a Tensor or NumPy Array: '{thetas}'")
        if not isinstance(nablas, (Tensor, np.ndarray)):
            # invalid datatype
            raise TypeError(f"'nablas' is not a Tensor or NumPy Array: '{nablas}'")

        # return updated thetas
        if isinstance(nablas, Tensor):
            # nablas conversion
            nablas = nablas.to_array()
        if isinstance(thetas, Tensor):
            # Tensor
            return Tensor(self._optim(thetas.to_array(), nablas))
        else:
            # numpy array
            return self._optim(thetas, nablas)
