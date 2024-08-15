r"""
**Algorithms for GardenPy.**

Attributes:
----------
**Initializers**:
    Initialization algorithms for kernels/weights/biases.
**Activators**:
    Activation algorithms for activations.
**Losses**:
    Loss algorithms for loss.
**Optimizers**:
    Optimization algorithms for kernels/weights/biases.


Notes:
----------
- Refer to GardenPy's repository or GardenPy's docs for more information.
"""

from typing import Union
import warnings

import numpy as np

from .objects import Tensor


class Initializers:
    r"""
    **Initialization algorithms for kernels/weights/biases.**

    These algorithms ideally support GardenPy Tensors, but are compatible with NumPy Arrays.

    Attributes:
    ----------
    **algorithm** : (*str*)
        Initialization algorithm.
    **parameters** : (*dict*)
        Parameters for initialization algorithm.

    Methods:
    ----------
    **__init__(algorithm: str, *, parameters: dict = None, **kwargs)** :
        Instantiates the object with the specified parameters.

    **initialize(self, rows: int, columns: int, *, tens=True) -> Union[Tensor, np.ndarray]** :
        Initializes an array with the specified dimensions.
        If *tens* is 'True', the initialized matrix will be a Tensor.

    Notes:
    ----------
    - Initializers supports GardenPy Tensors; however, Initializers also works with NumPy Arrays.

    - Refer to GardenPy's repository or GardenPy's docs for more information.
    """
    def __init__(self, algorithm: str, *, parameters: dict = None, **kwargs):
        r"""
        **Initializer initialization with defined parameters.**

        Parameters:
        ----------
        **algorithm** : (*str*) {'*xavier*', '*gaussian*', '*uniform*'}
            Initialization algorithm.

            - *xavier* : Xavier/Glorot.
            - *gaussian*: Gaussian/Normal.
            - *uniform*: Uniform values.

        **parameters** (*dict*, *optional*) :
            Parameters for Initialization algorithm.

            - **xavier** : (*dict*) {'*mu*', '*sigma*'}
                - *mu* : (float, int), default=0.0
                    Mean.
                - *sigma* : (float, int), default=1.0
                    Standard deviation.

            - **gaussian** : (*dict*) {'*mu*', '*sigma*'}
                - *mu* : (float, int), default=0.0
                    The mean.
                - *sigma* : (float, int), default=1.0
                    The standard deviation.

            - **uniform** : (*dict*) {'*value*'}
                - *value* : (float, int), default=1.0
                    The uniform value.

        ****kwargs** (*any*, *optional*) :
            The parameters for the initialization algorithm with keyword arguments if desired.

            To set a parameter, add a keyword argument that refers to one of the parameters.

        Notes:
        ----------
        - Any parameters not specified will be set to their default values.

        - Parameters that are specified but not used within the specified algorithm will be discarded.
            - The user will receive a warning when this occurs.

        - Initializers supports GardenPy Tensors; however, Initializers also works with NumPy Arrays.

        Example:
        -----
        >>> from gardenpy.utils.algorithms import Initializers
        >>> init = Initializers('gaussian', parameters={'mu': 1.0, 'sigma': 3.0})
        """
        # initialization algorithms
        self.algorithms = [
            'xavier',
            'gaussian',
            'uniform'
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
                'mu': 0.0,
                'sigma': 1.0
            },
            'gaussian': {
                'mu': 0.0,
                'sigma': 1.0
            },
            'uniform': {
                'value': 1.0
            }
        }
        # default initialization algorithm parameter datatypes
        dtypes = {
            'xavier': {
                'mu': (float, int),
                'sigma': (float, int)
            },
            'gaussian': {
                'mu': (float, int),
                'sigma': (float, int)
            },
            'uniform': {
                'value': (float, int),
            }
        }
        # default initialization algorithm parameter value types
        vtypes = {
            'xavier': {
                'mu': lambda x: True,
                'sigma': lambda x: True
            },
            'gaussian': {
                'mu': lambda x: True,
                'sigma': lambda x: True
            },
            'uniform': {
                'value': lambda x: True,
            }
        }
        # default initialization algorithm parameter conversion types
        ctypes = {
            'xavier': {
                'mu': lambda x: float(x),
                'sigma': lambda x: float(x)
            },
            'gaussian': {
                'mu': lambda x: float(x),
                'sigma': lambda x: float(x)
            },
            'uniform': {
                'value': lambda x: float(x),
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
            return self._params['sigma'] * np.sqrt(2.0 / float(row + col)) * np.random.randn(row, col) + self._params['mu']

        def gaussian(row, col):
            # Gaussian initialization
            return self._params['sigma'] * np.random.randn(row, col) + self._params['mu']

        def uniform(row, col):
            # Zeros uniform initialization
            return self._params['value'] * np.ones((row, col), dtype=np.float64)

        # initialization algorithm dictionary
        init_funcs = {
            'xavier': xavier,
            'gaussian': gaussian,
            'uniform': uniform,
        }

        # return initialization algorithm
        return init_funcs[self._algorithm]

    def initialize(self, rows: int, columns: int, *, tens=True) -> Union[Tensor, np.ndarray]:
        r"""
        **Initializes matrix with the specified dimensions.**

        If *tens* is True, the initialized matrix will be a Tensor.

        This function ideally supports GardenPy Tensors, but is compatible with NumPy Arrays.

        Parameters:
        ----------
        - **rows** : (*int*)
            The rows.
        - **columns** : (*int*)
            The columns.
        - **tens** : (*bool*), default=True
            Whether to initialize as a Tensor.

        Returns:
        ----------
        - **matrix** : (*Union[Tensor, np.ndarray]*)
            The initialized matrix.

        Notes:
        ----------
        - The initialized matrix will support automatic tracking for automatic differentiation if it is a Tensor.

        Example:
        ----------
        >>> from gardenpy.utils.algorithms import Initializers
        >>> init = Initializers('gaussian', parameters={'mu': 1.0, 'sigma': 3.0})
        >>> mat = init.initialize(3, 4)
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
    r"""
    **Activation algorithms for GardenPy.**

    These algorithms ideally support GardenPy Tensors, but are compatible with NumPy Arrays.
    With NumPy Arrays, there will be no tracking.

    Attributes:
    ----------
    **algorithm** : (*str*)
        The activation algorithm.
    **parameters** : (*dict*)
        The parameters for the activation algorithm.

    Methods:
    ----------
    **__init__(algorithm: str, *, parameters: dict = None, **kwargs)** :
        Instantiates the object with the specified parameters.

    **activate(self, x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]** :
        Activates values with the activation algorithm.
        The activated values will retain the same datatype as x.
        If x is a Tensor, Activators will automatically track the equation for automatic differentiation.

    **d_activate(self, x: np.ndarray) -> np.ndarray** :
        Calculates derivative of x based on the specified outputs.
        Only supports NumPy Arrays for manual calculation.
        Use 'nabla' from gardenpy.utils.operators for Tensor support.

    Notes:
    ----------
    - Activators supports automatic differentiation built-in with Tensors.
        - d_activate should never be called when using Tensors.
        - Use 'nabla' from gardenpy.utils.operators for Tensor support.

    - Refer to GardenPy's repository or GardenPy's docs for more information.
    """
    def __init__(self, algorithm: str, *, parameters: dict = None, **kwargs):
        r"""
        **Activator initialization with defined parameters.**

        Parameters:
        ----------
        **algorithm** : (*str*) {'*softmax*', '*relu*', '*lrelu*', '*sigmoid*', '*tanh*', '*softplus*', '*mish*'}
            The activator algorithm.

            - *softmax* : Softmax.
            - *relu*: Rectified Linear Unit.
            - *lrelu*: Leaky Rectified Linear Unit.
            - *sigmoid*: Sigmoid.
            - *tanh*: Tanh.
            - *softplus*: SoftPlus.
            - *mish*: Mish.

        **parameters** (*dict*, *optional*) :
            The parameters for the activation algorithm.

            - **lrelu** : (*dict*) {'*beta*'}
                - *beta* : (float, int, 0.0<=beta), default=1e-2
                    The negative slope.

            - **softplus** : (*dict*) {'*beta*'}
                - *beta* : (float, int, 0.0<=beta), default=1.0
                    The vertical stretch.

            - **mish** : (*dict*) {'*beta*'}
                - *beta* : (float, int, 0.0<=beta), default=1.0
                    The vertical stretch.

        ****kwargs** (*any*, *optional*) :
            The parameters for the activation algorithm with keyword arguments if desired.

            To set a parameter, add a keyword argument that refers to one of the parameters.

        Notes:
        ----------
        - Any parameters not specified will be set to their default values.

        - Parameters that are specified but not used within the specified algorithm will be discarded.
            - The user will receive a warning when this occurs.

        - Activators supports GardenPy Tensors; however, Activators also works with NumPy Arrays.

        Example:
        -----
        >>> from gardenpy.utils.algorithms import Activators
        >>> optim = Activators('relu')
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
                'beta': 1e-2
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
        # default activation algorithm parameter conversion types
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
            return (np.exp(-x) + 1.0) ** -1.0

        def tanh(x):
            # Tanh activation
            return np.tanh(x)

        def softplus(x):
            # Softplus activation
            return np.log(np.exp(self._params['beta'] * x) + 1.0) / self._params['beta']

        def mish(x):
            # Mish activation
            return x * np.tanh(np.log(np.exp(self._params['beta'] * x) + 1.0) / self._params['beta'])

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
            return (np.sum(np.exp(x)) * np.exp(x) - np.exp(2.0 * x)) / (np.sum(np.exp(x)) ** 2.0)

        def d_relu(x, _=None):
            # derivative of ReLU activation
            return np.where(x > 0.0, 1.0, 0.0)

        def d_lrelu(x, _=None):
            # derivative of Leaky ReLU activation
            return np.where(x > 0.0, 1.0, self._params['beta'])

        def d_sigmoid(x, _=None):
            # derivative of Sigmoid activation
            return np.exp(-x) / ((np.exp(-x) + 1.0) ** 2.0)

        def d_tanh(x, _=None):
            # derivative of Tanh activation
            return np.cosh(x) ** -2.0

        def d_softplus(x, _=None):
            # derivative of Softplus activation
            return self._params['beta'] * np.exp(self._params['beta'] * x) / (self._params['beta'] * np.exp(self._params['beta'] * x) + self._params['beta'])

        def d_mish(x, _=None):
            # derivative of Mish activation
            return np.tanh(np.log(np.exp(self._params['beta'] * x) + 1.0) / self._params['beta']) + x * (np.cosh(np.log(np.exp(self._params['beta'] * x) + 1.0) / self._params['beta']) ** -2.0) * (self._params['beta'] * np.exp(self._params['beta'] * x) / (self._params['beta'] * np.exp(self._params['beta'] * x) + self._params['beta']))

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

    @staticmethod
    def _chain(upstream, downstream, _=None):
        return upstream * downstream

    def activate(self, x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        r"""
        **Activates values with the activation algorithm.**

        This function ideally supports GardenPy Tensors, but is compatible with NumPy Arrays.
        If x is a Tensor, Activators will automatically track the equation for automatic differentiation.

        Parameters:
        ----------
        - **yhat** : (*Union[Tensor, np.ndarray]*)
            The initial values.

        Returns:
        ----------
        - **y** : (*Union[Tensor, np.ndarray]*)
            The activated values.

        Notes:
        ----------
        - y will retain the same datatype as x.
        - If x is a Tensor, Activators will automatically track the equation for automatic differentiation.

        Example:
        ----------
        >>> from gardenpy.utils.objects import Tensor
        >>> from gardenpy.utils.algorithms import Activators
        >>> act = Activators('relu')
        >>> in_value = Tensor([-0.5, 1])
        >>> out_value = act.activate(in_value)
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
            x.tracker['chn'].append(self._chain)
            # track origin
            result.tracker['org'] = [x, '_']
            # track relation
            x.tracker['rlt'].append(['_', result])
            # result.id = x.id + 1

            # return result
            return result
        else:
            # return result
            return self._activator(x)

    def d_activate(self, x: np.ndarray) -> np.ndarray:
        r"""
        **Calculates the gradient of x with respect to y.**

        Only supports NumPy Arrays for manual calculation.
        Use 'nabla' from gardenpy.utils.operators for Tensor support.

        Parameters:
        ----------
        - **x** : (*np.ndarray*)
            The initial values.

        Returns:
        ----------
        - **y** : (*np.ndarray*)
            The gradient of x with respect to y.

        Notes:
        ----------
        - Only supports NumPy Arrays.
            - For Tensor support, use 'nabla' from gardenpy.utils.operators.

        Example:
        ----------
        >>> from numpy import array
        >>> from gardenpy.utils.algorithms import Activators
        >>> act = Activators('relu')
        >>> in_value = array([0, 1])
        >>> out_value = act.activate(in_value)
        >>> grad_in = act.d_activate(in_value)
        """
        if not isinstance(x, np.ndarray):
            # invalid datatype
            raise TypeError(f"'x' is not a NumPy Array: '{x}'")

        # return numpy array
        return self._d_activator(x)


class Losses:
    r"""
    **Loss algorithms for GardenPy.**

    These algorithms ideally support GardenPy Tensors, but are compatible with NumPy Arrays.
    With NumPy Arrays, there will be no tracking.

    Attributes:
    ----------
    **algorithm** : (*str*)
        The loss algorithm.
    **parameters** : (*dict*)
        The parameters for the loss algorithm.

    Methods:
    ----------
    **__init__(algorithm: str, *, parameters: dict = None, **kwargs)** :
        Instantiates the object with the specified parameters.

    **loss(yhat: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]** :
        Calculates loss based on the specified outputs.
        The loss will retain the same datatype as yhat.
        If yhat is a Tensor, Losses will automatically track the equation for automatic differentiation.

    **d_loss(yhat: np.ndarray, y: np.ndarray) -> np.ndarray** :
        Calculates derivative of the loss based on the specified outputs.
        Only supports NumPy Arrays for manual calculation.
        Use 'nabla' from gardenpy.utils.operators for Tensor support.

    Notes:
    ----------
    - Losses supports automatic differentiation built-in with Tensors.
        - d_loss should never be called when using Tensors.
        - Use 'nabla' from gardenpy.utils.operators for Tensor support.

    - Losses supports GardenPy Tensors; however, Losses also works with NumPy Arrays.

    - Refer to GardenPy's repository or GardenPy's docs for more information.
    """
    def __init__(self, algorithm: str, *, parameters: dict = None, **kwargs):
        r"""
        **Loss initialization with defined parameters.**

        Parameters:
        ----------
        **algorithm** : (*str*) {'*centropy*', '*ssr*', '*savr*'}
            The loss algorithm.

            - *centropy* : Categorical Cross-Entropy.
            - *ssr*: Sum of the Squared Residuals.
            - *savr*: Sum of the Absolute Value Residuals.

        **parameters** (*dict*, *optional*) :
            The parameters for the loss algorithm.

            Currently, no loss algorithms accept parameters.

        ****kwargs** (*any*, *optional*) :
            The parameters for the loss algorithm with keyword arguments if desired.

            To set a parameter, add a keyword argument that refers to one of the parameters.

        Notes:
        ----------
        - Any parameters not specified will be set to their default values.

        - Parameters that are specified but not used within the specified algorithm will be discarded.
            - The user will receive a warning when this occurs.

        Example:
        -----
        >>> from gardenpy.utils.algorithms import Losses
        >>> optim = Losses('centropy')
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
            'focal': {
                'alpha': 1.0,
                'gamma': 2.0
            },
            'ssr': None,
            'savr': None
        }
        # default loss algorithm parameter datatypes
        dtypes = {
            'focal': {
                'alpha': (float, int),
                'gamma': (float, int)
            }
        }
        # default loss algorithm parameter value types
        vtypes = {
            'focal': {
                'alpha': lambda x: 0.0 < x <= 1.0,
                'gamma': lambda x: 0.0 <= x
            }
        }
        # default loss algorithm parameter conversion types
        ctypes = {
            'focal': {
                'alpha': lambda x: float(x),
                'gamma': lambda x: float(x)
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

    def _get_loss(self):
        # defines loss algorithm
        def centropy(yhat, y):
            # Cross-Entropy loss
            return -np.sum(y * np.log(yhat))

        def focal(yhat, y):
            return -np.sum(self._params['alpha'] * (y - yhat) ** self._params['gamma'] * np.log(yhat))

        def ssr(yhat, y):
            # Sum of the Squared Residuals loss
            return np.sum((y - yhat) ** 2.0)

        def savr(yhat, y):
            # Sum of the Absolute Valued Residuals loss
            return np.sum(np.abs(y - yhat))

        # loss algorithm dictionary
        loss_funcs = {
            'centropy': centropy,
            'focal': focal,
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

        def d_focal(yhat, y):
            return -1 * self._params["alpha"] * (y - yhat) ** self._params["gamma"] / yhat - self._params["alpha"] * self._params["gamma"] * (y - yhat) ** (self._params["gamma"] - 1.0) * np.log(yhat)


        def d_ssr(yhat, y):
            # derivative of Sum of the Squared Residuals loss
            return -2.0 * (y - yhat)

        def d_savr(yhat, y):
            # derivative of Sum of the Absolute Valued Residuals loss
            return -np.sign(y - yhat)

        # derivative of loss algorithm dictionary
        d_loss_funcs = {
            'centropy': d_centropy,
            'focal': d_focal,
            'ssr': d_ssr,
            'savr': d_savr
        }

        # return derivative of loss algorithm
        return d_loss_funcs[self._algorithm]

    @staticmethod
    def _chain(downstream, upstream, _=None):
        return downstream * upstream

    def loss(self, yhat: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        r"""
        **Calculates loss based on the specified outputs.**

        This function ideally supports GardenPy Tensors, but is compatible with NumPy Arrays.
        If yhat is a Tensor, loss will automatically track the equation for automatic differentiation.

        Parameters:
        ----------
        - **yhat** : (*Union[Tensor, np.ndarray]*)
            The predicted values.
        - **y** : (*Union[Tensor, np.ndarray]*)
            The expected values.

        Returns:
        ----------
        - **loss** : (*Union[Tensor, np.ndarray]*)
            The calculated loss.

        Notes:
        ----------
        - The loss will retain the same datatype as yhat.
        - If yhat is a Tensor, Losses will automatically track the equation for automatic differentiation.

        Example:
        ----------
        >>> from gardenpy.utils.objects import Tensor
        >>> from gardenpy.utils.algorithms import Losses
        >>> loss = Losses('ssr')
        >>> expected = Tensor([0, 1])
        >>> predicted = Tensor([0.2, 0.5])
        >>> cost = loss.loss(expected, predicted)
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
            yhat.tracker['chn'].append(self._chain)
            # track origin
            result.tracker['org'] = [yhat, y]
            # track relation
            yhat.tracker['rlt'].append([y, result])
            # result.id = yhat.id + 1

            # return result
            return result
        else:
            # return result
            return self._loss(yhat, y_arr)

    def d_loss(self, yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""
        **Calculates the gradient of loss with respect to yhat based on the specified outputs.**

        Only supports NumPy Arrays for manual calculation.
        Use 'nabla' from gardenpy.utils.operators for Tensor support.

        Parameters:
        ----------
        - **yhat** : (*np.ndarray*)
            The predicted values.
        - **y** : (*np.ndarray*)
            The expected values.

        Returns:
        ----------
        - **grad_yhat** : (*np.ndarray*)
            The gradient of loss with respect to yhat.

        Notes:
        ----------
        - Only supports NumPy Arrays.
            - For Tensor support, use 'nabla' from gardenpy.utils.operators.

        Example:
        ----------
        >>> from numpy import array
        >>> from gardenpy.utils.algorithms import Losses
        >>> loss = Losses('ssr')
        >>> expected = array([0, 1])
        >>> predicted = array([0.2, 0.5])
        >>> cost = loss.loss(expected, predicted)
        >>> d_cost = loss.d_loss(expected, predicted)
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
    r"""
    **Optimization algorithms for GardenPy.**

    These algorithms ideally support GardenPy Tensors, but are compatible with NumPy Arrays.

    Attributes:
    ----------
    **algorithm** : (*str*)
        The optimization algorithm.
    **hyperparameters** : (*dict*)
        The hyperparameters for the optimization algorithm.

    Methods:
    ----------
    **__init__(algorithm: str, *, hyperparameters: dict = None, **kwargs)** :
        Instantiates the object with the specified hyperparameters.

    **optimize(thetas: Union[Tensor, np.ndarray], nablas: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]** :
        Optimizes the thetas based on the specified gradients.
        The optimized thetas will retain the same datatype as the initial thetas.

    Notes:
    ----------
    - Optimizers optimizes thetas iteratively within each call.

    - Optimizers does not calculate gradients.
        - Gradients are calculated from GardenPy Tensors' automatic differentiation or by hand.

    - Optimizers supports GardenPy Tensors; however, Optimizers also works with NumPy Arrays.

    - Any values that must be called outside one instance of optimization are automatically saved to memory.
        - These values are saved within each object and called when necessary.
        - These values are not readily callable externally.

    - Refer to GardenPy's repository or GardenPy's docs for more information.
    """
    def __init__(self, algorithm: str, *, hyperparameters: dict = None, correlator=True, **kwargs):
        r"""
        **Optimizer initialization with defined hyperparameters.**

        Parameters:
        ----------
        **algorithm** : (*str*) {'*adam*', '*sgd*', '*rmsp*'}
            The optimization algorithm.

            - *adam* : Adaptive Moment Estimation.
            - *sgd*: Stochastic Gradient Descent.
            - *rmsp*: Root Mean Squared Propagation.

        **hyperparameters** (*dict*, *optional*) :
            The hyperparameters for the optimization algorithm.

            - **adam** : (*dict*) {'*alpha*', '*lambda_d*', '*beta1*', '*beta2*', '*epsilon*', '*ams*'}
                - *alpha* : (float, int), default=1e-3
                    Learning rate.
                - *lambda_d* : (float, int, 0.0 <= lambda_d), default=0.0
                    Strength of weight decay / L2 regularization.
                - *beta1* : (float, int, 0.0 <= beta1 < 1), default=0.9
                    Exponential decay rate for the first moment (mean).
                - *beta2* : (float, int, 0.0 <= beta2 < 1), default=0.999
                    Exponential decay rate for the second moment (uncentered variance).
                - *epsilon* : (float, int, 0.0 < epsilon), default=1e-8
                    Numerical stability constant to prevent division by zero.
                - *ams* : (bool), default=False
                    Whether to use AMSGrad.

            - **sgd** : (*dict*) {'*alpha*', '*lambda_d*', '*mu*', '*tau*', '*nesterov*'}
                - *alpha* : (float, int), default=1e-3
                    Learning rate.
                - *lambda_d* : (float, int, 0.0 <= lambda_d), default=0.0
                    Strength of weight decay / L2 regularization.
                - *mu* : (float, int, 0 <= mu < 1.0), default=0.0
                    Decay rate for the previous delta.
                - *tau* : (float, int, 0.0 <= tau < 1.0), default=0.0
                    Dampening of the current delta.
                - *nesterov* : (bool), default=False
                    Whether to use Nesterov momentum.

            - **rmsp** : (*dict*) {'*alpha*', '*lambda_d*', '*beta2*', '*mu*', '*epsilon*'}
                - *alpha* : (float, int), default=1e-3
                    Learning rate.
                - *lambda_d* : (float, int, 0.0 <= lambda_d), default=0.0
                    Strength of weight decay / L2 regularization.
                - *beta* : (float, int, 0.0 <= beta < 1), default=0.99
                    Exponential decay rate for the first moment (mean).
                - *mu* : (float, int, 0 <= mu < 1.0), default=0.0
                    Decay rate for the previous delta.
                - *epsilon* : (float, int, 0.0 < epsilon), default=1e-8
                    Numerical stability constant to prevent division by zero.

        ****kwargs** (*any*, *optional*) :
            The hyperparameters for the optimization algorithm with keyword arguments if desired.

            To set a hyperparameter, add a keyword argument that refers to one of the hyperparameters.

        Notes:
        ----------
        - Any hyperparameters not specified will be set to their default values.

        - Hyperparameters that are specified but not used within the specified algorithm will be discarded.
            - The user will receive a warning when this occurs.

        - The internal memory will automatically be initialized when Optimizers is instantiated.

        Example:
        -----
        >>> from gardenpy.utils.algorithms import Optimizers
        >>> optim = Optimizers('adam', hyperparameters={'alpha': 1e-2, 'lambda_d': 1e-3, 'ams': True})
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
        self._correlator = correlator
        if correlator:
            # instance memorization
            self._tensors = []
            self._full = []
            self._memory = None
        else:
            # set memorization
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
                'alpha': 1e-3,
                'lambda_d': 0.0,
                'beta1': 0.9,
                'beta2': 0.999,
                'epsilon': 1e-8,
                'ams': False
            },
            'sgd': {
                'alpha': 1e-3,
                'lambda_d': 0.0,
                'mu': 0.0,
                'tau': 0.0,
                'nesterov': False
            },
            'rmsp': {
                'alpha': 1e-3,
                'lambda_d': 0.0,
                'beta2': 0.99,
                'mu': 0.0,
                'epsilon': 1e-8
            },
            'adag': {
                'alpha': 1e-2,
                'lambda_d': 0.0,
                'tau': 0.0,
                'nu': 0.0,
                'epsilon': 1e-8
            }
        }
        # default optimization algorithm hyperparameter datatypes
        dtypes = {
            'adam': {
                'alpha': (float, int),
                'lambda_d': (float, int),
                'beta1': (float, int),
                'beta2': (float, int),
                'epsilon': (float, int),
                'ams': (bool, int)
            },
            'sgd': {
                'alpha': (float, int),
                'lambda_d': (float, int),
                'mu': (float, int),
                'tau': (float, int),
                'nesterov': (bool, int)
            },
            'rmsp': {
                'alpha': (float, int),
                'lambda_d': (float, int),
                'beta2': (float, int),
                'mu': (float, int),
                'epsilon': (float, int)
            },
            'adag': {
                'alpha': (float, int),
                'lambda_d': (float, int),
                'tau': (float, int),
                'nu': (float, int),
                'epsilon': (float, int)
            }
        }
        # default optimization algorithm hyperparameter value types
        vtypes = {
            'adam': {
                'alpha': lambda x: True,
                'lambda_d': lambda x: 0.0 <= x < 1.0,
                'beta1': lambda x: 0.0 <= x < 1.0,
                'beta2': lambda x: 0.0 <= x < 1.0,
                'epsilon': lambda x: 0.0 < x <= 1e-2,
                'ams': lambda x: True
            },
            'sgd': {
                'alpha': lambda x: True,
                'lambda_d': lambda x: 0.0 <= x < 1.0,
                'mu': lambda x: 0.0 <= x < 1.0,
                'tau': lambda x: 0.0 <= x < 1.0,
                'nesterov': lambda x: True
            },
            'rmsp': {
                'alpha': lambda x: True,
                'lambda_d': lambda x: 0.0 <= x < 1.0,
                'beta2': lambda x: 0.0 <= x < 1.0,
                'mu': lambda x: 0.0 <= x < 1.0,
                'epsilon': lambda x: 0.0 < x <= 1e-2
            },
            'adag': {
                'alpha': lambda x: True,
                'lambda_d': lambda x: 0.0 <= x < 1.0,
                'tau': lambda x: 0.0 <= x <= 1.0,
                'nu': lambda x: 0.0 <= x <= 1.0,
                'epsilon': lambda x: 0.0 < x <= 1e-2
            }
        }
        # default optimization algorithm hyperparameter conversion types
        ctypes = {
            'adam': {
                'alpha': lambda x: float(x),
                'lambda_d': lambda x: float(x),
                'beta1': lambda x: float(x),
                'beta2': lambda x: float(x),
                'epsilon': lambda x: float(x),
                'ams': lambda x: bool(x)
            },
            'sgd': {
                'alpha': lambda x: float(x),
                'lambda_d': lambda x: float(x),
                'mu': lambda x: float(x),
                'tau': lambda x: float(x),
                'nesterov': lambda x: bool(x)
            },
            'rmsp': {
                'alpha': lambda x: float(x),
                'lambda_d': lambda x: float(x),
                'beta2': lambda x: float(x),
                'mu': lambda x: float(x),
                'epsilon': lambda x: float(x)
            },
            'adag': {
                'alpha': lambda x: float(x),
                'lambda_d': lambda x: float(x),
                'tau': lambda x: float(x),
                'nu': lambda x: float(x),
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
                'iota': 1.0,
                'upsilons_hat_mx': None
            },
            'sgd': {
                'deltas_p': None
            },
            'rmsp': {
                'deltas_p': None,
                'upsilons_p': None
            },
            'adag': {
                'iota': 1.0,
                'tau': None
            }
        }
        # return memory dictionary
        return memories[self._algorithm]

    def _get_optim(self):
        # defines optimization algorithm
        def adam(thetas, nablas):
            # Adaptive Moment Estimation optimization algorithm
            if self._hyps['lambda_d']:
                # weight decay
                nablas = nablas + self._hyps['lambda_d'] * thetas

            if self._memory['deltas_p'] is not None:
                # zero-biased first moment estimate
                deltas = self._hyps['beta1'] * self._memory['deltas_p'] + (1.0 - self._hyps['beta1']) * nablas
            else:
                # zero-biased initialized first moment estimate
                deltas = (1.0 - self._hyps['beta1']) * nablas
            # zero-biased first moment memory update
            self._memory['deltas_p'] = deltas

            if self._memory['upsilons_p'] is not None:
                # zero-biased second raw moment estimate
                upsilons = self._hyps['beta2'] * self._memory['upsilons_p'] + (1.0 - self._hyps['beta2']) * nablas ** 2.0
            else:
                # zero-biased initialized second raw moment estimate
                upsilons = (1.0 - self._hyps['beta2']) * nablas ** 2.0
            # zero-biased second raw moment memory update
            self._memory['upsilons_p'] = upsilons

            # zero-bias correction
            deltas_hat = deltas / (1.0 - self._hyps['beta1'] ** self._memory['iota'])
            upsilons_hat = upsilons / (1.0 - self._hyps['beta2'] ** self._memory['iota'])
            # zero-bias factor update
            self._memory['iota'] += 1.0

            if self._hyps['ams']:
                # strongly non-convex decaying learning rate variant
                if self._memory['upsilons_hat_mx'] is not None:
                    # maximum zero-biased second raw moment estimate
                    upsilons_hat_mx = np.maximum(self._memory['upsilons_hat_mx'], upsilons_hat)
                else:
                    # initial maximum zero-biased second raw moment estimate
                    upsilons_hat_mx = np.maximum(0.0 * upsilons_hat, upsilons_hat)
                upsilons_hat = upsilons_hat_mx
                # maximum zero-biased second raw moment memory update
                self._memory['upsilons_hat_mx'] = upsilons_hat_mx

            # optimization
            thetas -= self._hyps['alpha'] * deltas_hat / (np.sqrt(upsilons_hat) + self._hyps['epsilon'])
            return thetas

        def sgd(thetas, nablas):
            # Stochastic Gradient Descent optimization algorithm
            if self._hyps['lambda_d']:
                # weight decay
                nablas = nablas + self._hyps['lambda_d'] * thetas

            if self._hyps['mu'] and (self._memory['deltas_p'] is not None):
                # momentum and dampening
                deltas = self._hyps['mu'] * self._memory['deltas_p'] + (1.0 - self._hyps['tau']) * nablas
                # delta memory update
                self._memory['deltas_p'] = deltas
            elif self._hyps['mu']:
                # dampening
                deltas = (1.0 - self._hyps['tau']) * nablas
                # delta memory update
                self._memory['deltas_p'] = deltas
            else:
                # dampening
                deltas = (1.0 - self._hyps['tau']) * nablas

            if self._hyps['nesterov'] and self._hyps['mu']:
                # nesterov variant
                deltas = nablas + self._hyps['mu'] * deltas

            # optimization
            thetas -= self._hyps['alpha'] * deltas
            return thetas

        def rmsp(thetas, nablas):
            # Root Mean Squared Propagation optimization algorithm
            if self._hyps['lambda_d']:
                # weight decay
                nablas = nablas + self._hyps['lambda_d'] * thetas

            if self._memory['upsilons_p'] is not None:
                # zero-biased second raw moment estimate
                upsilons = self._hyps['beta2'] * self._memory['upsilons_p'] + (1.0 - self._hyps['beta2']) * nablas ** 2.0
            else:
                # zero-biased initialized second raw moment estimate
                upsilons = (1.0 - self._hyps['beta2']) * nablas ** 2.0
            # zero-biased second raw moment memory update
            self._memory['upsilons_p'] = upsilons

            # delta calculation
            deltas = nablas / (np.sqrt(upsilons) + self._hyps['epsilon'])

            if self._hyps['mu'] and (self._memory['deltas_p'] is not None):
                # momentum
                deltas += self._hyps['mu'] * self._memory['deltas_p']
                # delta memory update
                self._memory['deltas_p'] = deltas
            elif self._hyps['mu']:
                # delta memory update
                self._memory['deltas_p'] = deltas

            # optimization
            thetas -= self._hyps['alpha'] * deltas
            return thetas

        def adag(thetas, nablas):
            # todo
            # Adaptive Gradient Algorithm optimization algorithm
            if self._hyps['lambda_d']:
                # weight decay
                nablas = nablas + self._hyps['lambda_d'] * thetas

            # learning rate adaption
            alpha_hat = self._hyps['alpha'] / ((self._memory['iota'] - 1.0) * self._hyps['nu'] + 1.0)
            # learning rate factor update
            self._memory['iota'] += 1.0

            if self._memory['tau'] is not None:
                # state-sum calculation
                tau = self._memory['tau'] + nablas ** 2.0
            else:
                # state-sum initialization
                tau = 0.0 * thetas + self._hyps['tau']
            # state-sum memory update
            self._memory['tau'] = tau

            # optimization
            thetas -= alpha_hat * nablas / (np.sqrt(tau) + self._hyps['epsilon'])
            return thetas

        # optimization algorithm dictionary
        optim_funcs = {
            'adam': adam,
            'sgd': sgd,
            'rmsp': rmsp,
            'adag': adag
        }

        # return optimization algorithm
        return optim_funcs[self._algorithm]

    def optimize(self, thetas: Union[Tensor, np.ndarray], nablas: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        r"""
        **Optimizes the thetas of a model based on the specified gradients.**

        This function ideally supports GardenPy Tensors, but is compatible with NumPy Arrays.

        Parameters:
        ----------
        - **thetas** : (*Union[Tensor, np.ndarray]*)
            The thetas.
        - **nablas** : (*Union[Tensor, np.ndarray]*)
            The gradients of the thetas.

        Returns:
        ----------
        - **thetas** : (*Union[Tensor, np.ndarray]*)
            The optimized thetas.

        Notes:
        ----------
        - The optimized thetas will retain the same datatype as the initial thetas.
        - The optimized thetas will support automatic tracking for automatic differentiation if it is a Tensor.

        - The internal memory will be accessed automatically upon the function call.

        Example:
        ----------
        >>> from gardenpy.utils.objects import Tensor
        >>> from gardenpy.utils.algorithms import Optimizers
        >>> optim = Optimizers('adam')
        >>> theta = Tensor([0, 1])
        >>> nabla = Tensor([0.2, 0.5])
        >>> theta = optim.optimize(theta, nabla)
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
            if self._correlator:
                if thetas.id in self._tensors:
                    # get memory instance
                    self._memory = self._full[self._tensors.index(thetas.id)]
                else:
                    # set memory instance
                    self._tensors.append(thetas.id)
                    self._full.append(self._get_memory())
                    self._memory = self._full[self._tensors.index(thetas.id)]
            # perform optimization
            result = Tensor(self._optim(thetas.to_array(), nablas))
            # maintain id
            result.id = thetas.id
            return result
        else:
            # numpy array
            result = self._optim(thetas, nablas)
            return result
