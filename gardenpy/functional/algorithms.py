r"""Implemented machine learning algorithms"""

from typing import Callable, List, Tuple, Optional, Union
import numpy as np

from .objects import Tensor
from utils.checkers import Params, ParamChecker


class Initializers:
    r"""
    **Initialization algorithms for weights and biases.**

    --------------------------------------------------------------------------------------------------------------------

    Creates GardenPy Tensors using the specified initialization method with the specified dimensions.
    It automatically creates GardenPy Tensors, but these can be converted into NumPy arrays if wanted.
    Initialization methods can be added with relative ease using the base structure provided within this class.
    """
    _methods: List[str] = [
        'xavier',
        'gaussian',
        'uniform'
    ]

    def __init__(self, method: str, *, hyperparameters: Optional[dict] = None, ikwiad: bool = False, **kwargs):
        r"""
        Sets internal initializer method and hyperparameters.
        Used for reference when Tensor creation is called.
        Any hyperparameters not set will set to their default value.

        xavier (Xavier/Glorot)
            - mu (float | int), default = 0.0: Distribution mean.
            - sigma (float | int), default = 1.0, 0.0 < sigma: Distribution standard deviation.
            - kappa (float | int), default = 1.0: Distribution gain.
        gaussian (Gaussian/Normal)
            - mu (float | int), default = 0.0: Distribution mean.
            - sigma (float | int), default = 1.0, 0.0 < sigma: Distribution standard deviation.
            - kappa (float | int), default = 1.0: Distribution gain.
        uniform (Uniform)
            - kappa (float | int), default = 1.0: Uniform value.

        Args:
            method (str): Initializer method.
            hyperparameters (dict, optional): Method hyperparameters.
            ikwiad (bool), default = False: Remove all warning messages ("I know what I am doing" - ikwiad).
            **kwargs: Alternate input format for method hyperparameters.

        Raises:
            TypeError: If any hyperparameters were of the wrong type.
            ValueError: If invalid values were passed for any of the hyperparameters.
        """
        # internals
        self._ikwiad = bool(ikwiad)
        self._method, self._hyperparams = self._get_method(method=method, hyperparams=hyperparameters, **kwargs)

        # set method
        self._func = None
        self._set_initializer()

    @classmethod
    def methods(cls) -> list:
        return cls._methods

    def _get_method(self, method: str, hyperparams: dict, **kwargs) -> Tuple[str, dict]:
        # hyperparameter reference
        default_hyperparams = {
            'xavier': Params(
                default={'mu': 0.0, 'sigma': 1.0, 'kappa': 1.0},
                dtypes={'mu': (float, int), 'sigma': (float, int), 'kappa': (float, int)},
                vtypes={'mu': lambda x: True, 'sigma': lambda x: 0 < x, 'kappa': lambda x: True},
                ctypes={'mu': lambda x: float(x), 'sigma': lambda x: float(x), 'kappa': lambda x: float(x)}
            ),
            'gaussian': Params(
                default={'mu': 0.0, 'sigma': 1.0, 'kappa': 1.0},
                dtypes={'mu': (float, int), 'sigma': (float, int), 'kappa': (float, int)},
                vtypes={'mu': lambda x: True, 'sigma': lambda x: 0 < x, 'kappa': lambda x: True},
                ctypes={'mu': lambda x: float(x), 'sigma': lambda x: float(x), 'kappa': lambda x: float(x)}
            ),
            'uniform': Params(
                default={'kappa': 1.0},
                dtypes={'kappa': float},
                vtypes={'kappa': lambda x: True},
                ctypes={'kappa': lambda x: float(x)}
            )
        }

        # check method
        if not isinstance(method, str):
            raise TypeError("'method' must be a string")
        if method not in Initializers._methods:
            raise ValueError(
                f"Invalid method: {method}\n"
                f"Choose from: {Initializers._methods}"
            )

        # set checker
        checker = ParamChecker(
            prefix=f'{method} hyperparameters',
            parameters=default_hyperparams[method],
            ikwiad=self._ikwiad
        )

        # return hyperparameters
        return method, checker.check_params(params=hyperparams, **kwargs)

    def _set_initializer(self) -> None:
        # hyperparameter reference
        if self._hyperparams is not None:
            h = self._hyperparams.copy()

        def initializer_method(func: Callable) -> Callable:
            def wrapper(*args: int) -> 'Tensor':
                # check dimensions
                if len(args) != 2:
                    raise ValueError("Initialization must occur with 2 dimensions")
                if not all(isinstance(arg, int) and 0 < arg for arg in args):
                    raise ValueError("Each dimension must be a positive integer")
                # initialize tensor
                return Tensor(func(*args))

            return wrapper

        @initializer_method
        def xavier(*args: int) -> np.ndarray:
            # xavier method function
            return (
                    h['kappa'] *
                    np.sqrt(2.0 / float(args[-2] + args[-1])) *
                    np.random.normal(loc=h['mu'], scale=h['sigma'], size=args)
            )

        @initializer_method
        def gaussian(*args: int) -> np.ndarray:
            # gaussian method function
            return h['kappa'] * np.random.normal(loc=h['mu'], scale=h['sigma'], size=args)

        @initializer_method
        def uniform(*args: int) -> np.ndarray:
            # uniform method function
            return h['kappa'] * np.ones(args, dtype=np.float64)

        # function reference
        funcs = {
            'xavier': xavier,
            'gaussian': gaussian,
            'uniform': uniform
        }
        self._func = funcs[self._method]

        # set initialize function
        def initialize(*args: int) -> Tensor:
            r"""
            Returns initialized Tensor with specified dimensions.

            Args:
                *args: Tensor's two dimensions of positive integers.

            Returns:
                Tensor: Initialized Tensor.

            Raises:
                ValueError: If the dimensions weren't properly set.
            """
            return self._func(*args)

        self.initialize = initialize


class Activators:
    r"""
    Activation algorithms for arrays.

    Applies an activation function to arrays.
    If used with GardenPy's Tensors, activation function utilizes the autograd methods, allowing for automatic gradient
    calculation and tracking with the activation functions.
    These activation functions can be used with NumPy arrays, but will lose automatic gradient tracking methods.
    The derivative of these activation functions can be called if using NumPy arrays, but function calls will go through
    no checks.
    If using Tensors, the derivatives will be automatically utilized during a nabla call.
    Activation methods can be added with relative ease using the base structure provided within this class.
    """
    _methods: List[str] = [
        'softmax',
        'relu',
        'lrelu',
        'sigmoid',
        'tanh',
        'softplus',
        'mish'
    ]

    def __init__(self, method: str, *, hyperparameters: Optional[dict] = None, ikwiad: bool = False, **kwargs):
        r"""
        Sets internal activator method and hyperparameters.
        Used for reference when activation is called.
        Any hyperparameters not set will set to their default value.

        softmax (Softmax)
            - None
        relu (ReLU)
            - None
        lrelu (Leaky ReLU)
            - beta (float | int), default = 1e-2, 0.0 < beta: Negative slope.
        sigmoid (Sigmoid)
            - None
        tanh (Tanh)
            - None
        softplus (Softplus)
            - beta (float | int), default = 1.0, 0.0 <= beta: Vertical stretch.
        mish (Mish)
            - beta (float | int), default = 1.0, 0.0 <= beta: Vertical stretch.

        Args:
            method (str): Activator method.
            hyperparameters (dict, optional): Method hyperparameters.
            ikwiad (bool), default = False: Remove all warning messages ("I know what I am doing" - ikwiad).
            **kwargs: Alternate input format for method hyperparameters.

        Raises:
            TypeError: If any hyperparameters were of the wrong type.
            ValueError: If invalid values were passed for any of the hyperparameters.
        """
        # allowed methods
        self._possible_methods = Activators._methods

        # internals
        self._ikwiad = bool(ikwiad)
        self._method, self._hyperparams = self._get_method(method=method, hyperparams=hyperparameters, **kwargs)

        # set method
        self._op = None
        self._set_activator()

    @classmethod
    def methods(cls) -> list:
        return cls._methods

    def _get_method(self, method: str, hyperparams: dict, **kwargs) -> Tuple[str, dict]:
        # hyperparameter reference
        default_hyperparams = {
            'softmax': Params(
                default=None,
                dtypes=None,
                vtypes=None,
                ctypes=None
            ),
            'relu': Params(
                default=None,
                dtypes=None,
                vtypes=None,
                ctypes=None
            ),
            'lrelu': Params(
                default={'beta': 1e-02},
                dtypes={'beta': (float, int)},
                vtypes={'beta': lambda x: x < 0},
                ctypes={'beta': lambda x: float(x)}
            ),
            'sigmoid': Params(
                default=None,
                dtypes=None,
                vtypes=None,
                ctypes=None
            ),
            'softplus': Params(
                default={'beta': 1.0},
                dtypes={'beta': (float, int)},
                vtypes={'beta': lambda x: x < 0},
                ctypes={'beta': lambda x: float(x)}
            ),
            'mish': Params(
                default={'beta': 1.0},
                dtypes={'beta': (float, int)},
                vtypes={'beta': lambda x: x < 0},
                ctypes={'beta': lambda x: float(x)}
            )
        }

        # check method
        if not isinstance(method, str):
            raise TypeError("'method' must be a string")
        if method not in Activators._methods:
            raise ValueError(
                f"Invalid method: {method}\n"
                f"Choose from: {Activators._methods}"
            )

        # set checker
        checker = ParamChecker(
            prefix=f'{method} hyperparameters',
            parameters=default_hyperparams[method],
            ikwiad=self._ikwiad
        )

        # return hyperparameters
        return method, checker.check_params(params=hyperparams, **kwargs)

    def _set_activator(self) -> None:
        # hyperparameter reference
        if self._hyperparams is not None:
            h = self._hyperparams.copy()

        class _Softmax(Tensor.LoneTensorMethod):
            r"""Softmax built-in method."""

            def __init__(self):
                super().__init__(prefix="softmax")

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return np.exp(x) / np.sum(np.exp(x))

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return (
                    (np.sum(np.exp(x)) * np.exp(x) - np.exp(2.0 * x)) /
                    (np.sum(np.exp(x)) ** 2.0) * np.eye(x.shape[1])
                )

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                # todo: incorrect math
                return down @ up

            def __call__(self, main: Tensor) -> Tensor:
                return self.call(main)

        class _ReLU(Tensor.LoneTensorMethod):
            r"""ReLU built-in method."""

            def __init__(self):
                super().__init__(prefix="relu")

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return np.maximum(0.0, x)

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return np.where(x > 0.0, 1.0, 0.0) * np.eye(x.shape[1])

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                return down @ up

            def __call__(self, main: Tensor) -> Tensor:
                return self.call(main)

        class _LeakyReLU(Tensor.LoneTensorMethod):
            r"""LeakyReLU built-in method."""

            def __init__(self):
                super().__init__(prefix="lrelu")

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return np.maximum(h['beta'] * x, x)

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return np.where(x > 0.0, 1.0, h['beta']) * np.eye(x.shape[1])

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                return down @ up

            def __call__(self, main: Tensor) -> Tensor:
                return self.call(main)

        class _Sigmoid(Tensor.LoneTensorMethod):
            r"""Sigmoid built-in method."""

            def __init__(self):
                super().__init__(prefix="sigmoid")

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return (np.exp(-x) + 1.0) ** -1.0

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return np.exp(-x) / ((np.exp(-x) + 1.0) ** 2.0) * np.eye(x.shape[1])

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                return down @ up

            def __call__(self, main: Tensor) -> Tensor:
                return self.call(main)

        class _Tanh(Tensor.LoneTensorMethod):
            r"""Tanh built-in method."""

            def __init__(self):
                super().__init__(prefix="tanh")

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return np.tanh(x)

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return np.cosh(x) ** -2.0 * np.eye(x.shape[1])

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                return down @ up

            def __call__(self, main: Tensor) -> Tensor:
                return self.call(main)

        class _Softplus(Tensor.LoneTensorMethod):
            r"""Softplus built-in method."""
            def __init__(self):
                super().__init__(prefix="softplus")

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return np.log(np.exp(h['beta'] * x) + 1.0) / h['beta']

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return (
                    h['beta'] * np.exp(h['beta'] * x) /
                    (h['beta'] * np.exp(h['beta'] * x) + h['beta']) *
                    np.eye(x.shape[1])
                )

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                return down @ up

            def __call__(self, main: Tensor) -> Tensor:
                return self.call(main)

        class _Mish(Tensor.LoneTensorMethod):
            r"""Mish built-in method."""

            def __init__(self):
                super().__init__(prefix="mish")

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return x * np.tanh(np.log(np.exp(h['beta'] * x) + 1.0) / h['beta'])

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return (
                    (
                        np.tanh(np.log(np.exp(h['beta'] * x) + 1.0) / h['beta']) +
                        x * (np.cosh(np.log(np.exp(h['beta'] * x) + 1.0) / h['beta']) ** -2.0) *
                        (h['beta'] * np.exp(h['beta'] * x) / (h['beta'] * np.exp(h['beta'] * x) + h['beta']))
                    ) * np.eye(x.shape[1])
                )  # aughhhhhh!!!

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                return down @ up

            def __call__(self, main: Tensor) -> Tensor:
                return self.call(main)

        # operator reference
        ops = {
            'softmax': _Softmax,
            'relu': _ReLU,
            'lrelu': _LeakyReLU,
            'sigmoid': _Sigmoid,
            'tanh': _Tanh,
            'softplus': _Softplus,
            'mish': _Mish
        }
        self._op = ops[self._method]

        # set method functions
        def activate(x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
            r"""
            for tensor or numpy
            """
            if isinstance(x, Tensor) and x.type == 'mat':
                return self._op(x)
            elif isinstance(x, np.ndarray):
                return self._op.forward(x)
            else:
                raise TypeError("'x' must be a Tensor with the matrix type or a NumPy array")

        def d_activate(x: np.ndarray) -> np.ndarray:
            r"""
            only for np
            """
            if not isinstance(x, np.ndarray):
                raise TypeError("'x' must be a NumPy array")
            return self._op.backward(x)

        self.activate = activate
        self.d_activate = d_activate


class Losses:
    _methods: List[str] = [
        'centropy',
        'ssr',
        'savr'
    ]

    def __init__(self, method: str, *, hyperparameters: Optional[dict] = None, ikwiad: bool = False, **kwargs):
        # allowed methods
        self._possible_methods = Losses._methods

        # internals
        self._ikwiad = bool(ikwiad)
        self._method, self._hyperparams = self._get_method(method=method, hyperparams=hyperparameters, **kwargs)

        # set method
        self._op = None
        self._set_loss()

    @classmethod
    def methods(cls):
        return cls._methods

    def _get_method(self, method: str, hyperparams: dict, **kwargs) -> Tuple[str, dict]:
        # hyperparameter reference
        default_hyperparams = {
            'centropy': Params(
                default={'epsilon': 1e-10},
                dtypes={'epsilon': float},
                vtypes={'epsilon': lambda x: 0.0 < x < 1.0},
                ctypes={'epsilon': lambda x: x}
            ),
            'ssr': Params(
                default=None,
                dtypes=None,
                vtypes=None,
                ctypes=None
            ),
            'savr': Params(
                default=None,
                dtypes=None,
                vtypes=None,
                ctypes=None
            ),
        }

        # check method
        if not isinstance(method, str):
            raise TypeError("'method' must be a string")
        if method not in Losses._methods:
            raise ValueError(
                f"Invalid method: {method}\n"
                f"Choose from: {Losses._methods}"
            )

        # set checker
        checker = ParamChecker(
            prefix=f'{method} hyperparameters',
            parameters=default_hyperparams[method],
            ikwiad=self._ikwiad
        )

        # return hyperparameters
        return method, checker.check_params(params=hyperparams, **kwargs)

    def _set_loss(self) -> None:
        # hyperparameter reference
        if self._hyperparams is not None:
            h = self._hyperparams.copy()

        class _CrossEntropy(Tensor.PairedTensorMethod):
            r"""Cross Entropy built-in method."""
            def __init__(self):
                super().__init__(prefix="centropy")

            @staticmethod
            def forward(yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
                return np.array([[-np.sum(y * np.log(yhat + h['epsilon']))]])

            @staticmethod
            def backward(yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
                return -y / (yhat + h['epsilon'])

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                return down @ up

            def __call__(self, main: Tensor, other: Tensor) -> Tensor:
                return self.call(main, other)

        class _SumOfSquaredResiduals(Tensor.PairedTensorMethod):
            r"""Sum of the squared residuals built-in method."""
            def __init__(self):
                super().__init__(prefix="ssr")

            @staticmethod
            def forward(yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
                return np.array([[np.sum((y - yhat) ** 2.0)]])

            @staticmethod
            def backward(yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
                return -2.0 * (y - yhat)

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                try:
                    return down @ up
                except ValueError:
                    return down * up

            def __call__(self, main: Tensor, other: Tensor) -> Tensor:
                return self.call(main, other)

        class _SumOfAbsoluteValueResiduals(Tensor.PairedTensorMethod):
            r"""Sum of the absolute value residuals built-in method."""
            def __init__(self):
                super().__init__(prefix="savr")

            @staticmethod
            def forward(yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
                return np.array([[np.sum(np.abs(y - yhat))]])

            @staticmethod
            def backward(yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
                return -np.sign(y - yhat)

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                return down @ up

            def __call__(self, main: Tensor, other: Tensor) -> Tensor:
                return self.call(main, other)

        # operator reference
        ops = {
            'centropy': _CrossEntropy,
            'ssr': _SumOfSquaredResiduals,
            'savr': _SumOfAbsoluteValueResiduals
        }
        self._op = ops[self._method]()

        # set method functions
        def loss(yhat: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
            r"""
            for tensor or numpy
            """
            if isinstance(yhat, Tensor) and yhat.type == 'mat':
                return self._op(yhat, y)
            elif isinstance(yhat, np.ndarray):
                return self._op.forward(yhat, y)
            else:
                raise TypeError("'y' must be a Tensor with the matrix type or an array")

        def d_loss(yhat: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray]) -> np.ndarray:
            r"""
            only for np
            """
            if not isinstance(yhat, np.ndarray):
                raise TypeError("'yhat' must be an array")
            if not isinstance(y, np.ndarray):
                raise TypeError("'y' must be an array")
            return self._op.backward(y, yhat)

        self.loss = loss
        self.d_loss = d_loss


class Optimizers:
    _methods: List[str] = [
        'adam',
        'sgd',
        'rmsp'
    ]

    def __init__(
            self,
            method: str,
            *,
            hyperparameters: Optional[dict] = None,
            correlator: bool = True,
            ikwiad: bool = False,
            **kwargs
    ):
        # internals
        self._ikwiad = bool(ikwiad)
        self._correlator = bool(correlator)
        self._method, self._hyperparams = self._get_method(method=method, hyperparams=hyperparameters, **kwargs)
        if self._correlator:
            self._memories = {}
        else:
            self._memories = None

        # set method
        self._alg = None
        self._set_optimizer()

    @classmethod
    def methods(cls) -> list:
        return cls._methods

    def _get_method(self, method: str, hyperparams: dict, **kwargs) -> Tuple[str, dict]:
        # hyperparameter reference
        default_hyperparams = {
            'adam': Params(
                default={
                    'alpha': 1e-03,
                    'lambda_d': 0.0,
                    'beta_1': 0.9,
                    'beta_2': 0.999,
                    'epsilon': 1e-10,
                    'ams': False
                },
                dtypes={
                    'alpha': (float, int),
                    'lambda_d': (float, int),
                    'beta_1': (float, int),
                    'beta_2': (float, int),
                    'epsilon': (float, int),
                    'ams': (bool, int)
                },
                vtypes={
                    'alpha': lambda x: True,
                    'lambda_d': lambda x: 0.0 <= x < 1.0,
                    'beta_1': lambda x: 0.0 < x < 1.0,
                    'beta_2': lambda x: 0.0 < x < 1.0,
                    'epsilon': lambda x: 0.0 < x <= 1e-02,
                    'ams': lambda x: True
                },
                ctypes={
                    'alpha': lambda x: float(x),
                    'lambda_d': lambda x: float(x),
                    'beta_1': lambda x: float(x),
                    'beta_2': lambda x: float(x),
                    'epsilon': lambda x: float(x),
                    'ams': lambda x: bool(x)
                }
            ),
            'sgd': Params(
                default={
                    'alpha': 1e-03,
                    'lambda_d': 0.0,
                    'mu': 0.0,
                    'tau': 0.0,
                    'nesterov': False
                },
                dtypes={
                    'alpha': (float, int),
                    'lambda_d': (float, int),
                    'mu': (float, int),
                    'tau': (float, int),
                    'nesterov': (bool, int)
                },
                vtypes={
                    'alpha': lambda x: True,
                    'lambda_d': lambda x: 0.0 <= x < 1.0,
                    'mu': lambda x: 0.0 <= x < 1.0,
                    'tau': lambda x: 0.0 <= x < 1.0,
                    'nesterov': lambda x: True
                },
                ctypes={
                    'alpha': lambda x: float(x),
                    'lambda_d': lambda x: float(x),
                    'mu': lambda x: float(x),
                    'tau': lambda x: float(x),
                    'nesterov': lambda x: bool(x)
                }
            ),
            'rmsp': Params(
                default={
                    'alpha': 1e-03,
                    'lambda_d': 0.0,
                    'beta': 0.99,
                    'mu': 0.0,
                    'epsilon': 1e-10
                },
                dtypes={
                    'alpha': (float, int),
                    'lambda_d': (float, int),
                    'beta': (float, int),
                    'mu': (float, int),
                    'epsilon': (float, int)
                },
                vtypes={
                    'alpha': lambda x: True,
                    'lambda_d': lambda x: 0.0 <= x < 1.0,
                    'beta': lambda x: 0.0 <= x < 1.0,
                    'mu': lambda x: 0.0 <= x < 1.0,
                    'epsilon': lambda x: 0.0 < x <= 1e-02
                },
                ctypes={
                    'alpha': lambda x: float(x),
                    'lambda_d': lambda x: float(x),
                    'beta': lambda x: float(x),
                    'mu': lambda x: float(x),
                    'epsilon': lambda x: float(x)
                }
            ),
            'adag': Params(
                default={
                    'alpha': 1e-03,
                    'lambda_d': 0.0,
                    'nu': 0.0,
                    'epsilon': 1e-10
                },
                dtypes={
                    'alpha': (float, int),
                    'lambda_d': (float, int),
                    'nu': (float, int),
                    'epsilon': (float, int)
                },
                vtypes={
                    'alpha': lambda x: True,
                    'lambda_d': lambda x: 0.0 <= x < 1.0,
                    'nu': lambda x: 0.0 <= x <= 1.0,
                    'epsilon': lambda x: 0.0 < x <= 1e-02
                },
                ctypes={
                    'alpha': lambda x: float(x),
                    'lambda_d': lambda x: float(x),
                    'nu': lambda x: float(x),
                    'epsilon': lambda x: float(x)
                }
            )
        }

        # check method
        if not isinstance(method, str):
            raise TypeError("'method' must be a string")
        if method not in Optimizers._methods:
            raise ValueError(
                f"Invalid method: {method}\n"
                f"Choose from: {Optimizers._methods}"
            )

        # set checker
        checker = ParamChecker(
            prefix=f'{method} hyperparameters',
            parameters=default_hyperparams[method],
            ikwiad=self._ikwiad
        )

        # return hyperparameters
        return method, checker.check_params(params=hyperparams, **kwargs)

    def _get_memories(self, theta: np.ndarray) -> dict:
        # instantiates memory dictionary
        memories = {
            'adam': {
                'psi_p': 0.0 * theta,
                'omega_p': 0.0 * theta,
                'iota': 1.0,
                'omega_hat_max': 0.0 * theta
            },
            'sgd': {
                'delta_p': 0.0 * theta
            },
            'rmsp': {
                'delta_p': 0.0 * theta,
                'omega_p': 0.0 * theta
            },
            'adag': {
                'omega_p': 0.0 * theta,
                'iota': 1.0
            }
        }
        # return memory dictionary
        return memories[self._method]

    def _set_optimizer(self) -> None:
        # hyperparameter reference
        if self._hyperparams is not None:
            h = self._hyperparams.copy()

        def adam(theta: np.ndarray, nabla: np.ndarray, m: dict) -> np.ndarray:
            gamma = nabla + h['lambda_d'] * theta

            psi = h['beta_1'] * m['psi_p'] + (1.0 - h['beta_1']) * gamma
            omega = h['beta_2'] * m['omega_p'] + (1.0 - h['beta_2']) * gamma ** 2.0
            m['psi_p'] = psi
            m['omega_p'] = omega

            psi_hat = psi / (1.0 - h['beta_1'] ** m['iota'])
            omega_hat = omega / (1.0 - h['beta_2'] ** m['iota'])

            if h['ams']:
                m['omega_hat_max'] = np.maximum(omega_hat, m['omega_hat_max'])
                omega_hat = m['omega_hat_max']

            m['iota'] += 1.0
            return h['alpha'] * psi_hat / (np.sqrt(omega_hat) + h['epsilon'])

        def sgd(theta: np.ndarray, nabla: np.ndarray, m: dict) -> np.ndarray:
            gamma = nabla + h['lambda_d'] * theta

            delta = h['mu'] * m['delta_p'] + (1.0 - h['tau']) * gamma

            if h['nesterov']:
                delta = h['mu'] * delta + gamma
            m['delta_p'] = delta

            return h['alpha'] * delta

        def rmsp(theta: np.ndarray, nabla: np.ndarray, m: dict) -> np.ndarray:
            gamma = nabla + h['lambda_d'] * theta

            omega = h['beta'] * m['omega_p'] + (1.0 - h['beta']) * gamma ** 2.0
            m['omega_p'] = omega

            delta = h['mu'] * m['delta_p'] + gamma / (np.sqrt(omega) + h['epsilon'])
            m['delta_p'] = delta

            return h['alpha'] * delta

        def adag(theta: np.ndarray, nabla: np.ndarray, m: dict) -> np.ndarray:
            gamma = nabla + h['lambda_d'] * theta

            alpha_hat = h['alpha'] / (1.0 + (m['iota'] - 1.0) * h['nu'])

            omega = m['omega_p'] + gamma ** 2.0
            m['omega_p'] = omega

            m['iota'] += 1.0
            return alpha_hat * gamma / (np.sqrt(omega) + h['epsilon'])

        # method reference
        algs = {
            'adam': adam,
            'sgd': sgd,
            'rmsp': rmsp,
            'adag': adag
        }
        # get method
        self._alg = algs[self._method]

        # set initialize function
        def optimize(theta: Union[Tensor, np.ndarray], nabla: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
            if isinstance(nabla, Tensor):
                nabla = nabla.array
            elif not isinstance(nabla, np.ndarray):
                raise TypeError("'nabla' must be a Tensor or array")

            if isinstance(theta, Tensor) and self._correlator:
                t_id = theta.id
                if t_id not in self._memories.keys():
                    self._memories.update({f'{t_id}': self._get_memories(theta=theta.array)})
                result = Tensor(self._alg(theta=theta.array, nabla=nabla, m=self._memories[f'{t_id}']))
                Tensor.instance_replace(replaced=t_id, replacer=result.id)
                return result
            elif isinstance(theta, Tensor):
                t_id = theta.id
                if self._memories is None:
                    self._memories = self._get_memories(theta=theta.array)
                result = Tensor(self._alg(theta=theta.array, nabla=nabla, m=self._memories))
                Tensor.instance_replace(replaced=t_id, replacer=result.id)
                return result
            elif isinstance(theta, np.ndarray) and not self._correlator:
                if self._memories is None:
                    self._memories = self._get_memories(theta=theta)
                return self._alg(theta=theta, nabla=nabla, m=self._memories)
            else:
                raise ValueError("Turn off correlator if using arrays")

        self.optimize = optimize


# this is here until I write the docstrings for the new optimizers class

# class OptimizersOld:
#     r"""
#     **Optimization algorithms for GardenPy.**
#
#     These algorithms ideally support GardenPy Tensors, but are compatible with NumPy Arrays.
#
#     Attributes:
#     ----------
#     **algorithm** : (*str*)
#         The optimization algorithm.
#     **hyperparameters** : (*dict*)
#         The hyperparameters for the optimization algorithm.
#
#     Methods:
#     ----------
#     **__init__(algorithm: str, *, hyperparameters: dict = None, **kwargs)** :
#         Instantiates the object with the specified hyperparameters.
#
#     **optimize(thetas: Union[Tensor, np.ndarray], nablas: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]** :
#         Optimizes the thetas based on the specified gradients.
#         The optimized thetas will retain the same datatype as the initial thetas.
#
#     Notes:
#     ----------
#     - Optimizers optimizes thetas iteratively within each call.
#
#     - Optimizers does not calculate gradients.
#         - Gradients are calculated from GardenPy Tensors' automatic differentiation or by hand.
#
#     - Optimizers supports GardenPy Tensors; however, Optimizers also works with NumPy Arrays.
#
#     - Any values that must be called outside one instance of optimization are automatically saved to memory.
#         - These values are saved within each object and called when necessary.
#         - These values are not readily callable externally.
#
#     - Refer to GardenPy's repository or GardenPy's docs for more information.
#     """
#
#     def __init__(self, algorithm: str, *, hyperparameters: dict = None, correlator=True, **kwargs):
#         r"""
#         **Optimizer initialization with defined hyperparameters.**
#
#         Parameters:
#         ----------
#         **algorithm** : (*str*) {'*adam*', '*sgd*', '*rmsp*', '*adag'}
#             The optimization algorithm.
#
#             - *adam* : Adaptive Moment Estimation.
#             - *sgd*: Stochastic Gradient Descent.
#             - *rmsp*: Root Mean Squared Propagation.
#             - *adag*: Adaptive Gradient Algorithm.
#
#         **hyperparameters** (*dict*, *optional*) :
#             The hyperparameters for the optimization algorithm.
#
#             - **adam** : (*dict*) {'*alpha*', '*lambda_d*', '*beta1*', '*beta2*', '*epsilon*', '*ams*'}
#                 - *alpha* : (float, int), default=1e-3
#                     Learning rate.
#                 - *lambda_d* : (float, int, 0.0 <= lambda_d), default=0.0
#                     Strength of weight decay / L2 regularization.
#                 - *beta1* : (float, int, 0.0 <= beta1 < 1), default=0.9
#                     Exponential decay rate for the first moment (mean).
#                 - *beta2* : (float, int, 0.0 <= beta2 < 1), default=0.999
#                     Exponential decay rate for the second moment (uncentered variance).
#                 - *epsilon* : (float, int, 0.0 < epsilon), default=1e-8
#                     Numerical stability constant to prevent division by zero.
#                 - *ams* : (bool), default=False
#                     Whether to use AMSGrad.
#
#             - **sgd** : (*dict*) {'*alpha*', '*lambda_d*', '*mu*', '*tau*', '*nesterov*'}
#                 - *alpha* : (float, int), default=1e-3
#                     Learning rate.
#                 - *lambda_d* : (float, int, 0.0 <= lambda_d), default=0.0
#                     Strength of weight decay / L2 regularization.
#                 - *mu* : (float, int, 0 <= mu < 1.0), default=0.0
#                     Decay rate for the previous delta.
#                 - *tau* : (float, int, 0.0 <= tau < 1.0), default=0.0
#                     Dampening of the current delta.
#                 - *nesterov* : (bool), default=False
#                     Whether to use Nesterov momentum.
#
#             - **rmsp** : (*dict*) {'*alpha*', '*lambda_d*', '*beta2*', '*mu*', '*epsilon*'}
#                 - *alpha* : (float, int), default=1e-3
#                     Learning rate.
#                 - *lambda_d* : (float, int, 0.0 <= lambda_d), default=0.0
#                     Strength of weight decay / L2 regularization.
#                 - *beta* : (float, int, 0.0 <= beta < 1), default=0.99
#                     Exponential decay rate for the first moment (mean).
#                 - *mu* : (float, int, 0 <= mu < 1.0), default=0.0
#                     Decay rate for the previous delta.
#                 - *epsilon* : (float, int, 0.0 < epsilon), default=1e-8
#                     Numerical stability constant to prevent division by zero.
#
#         ****kwargs** (*any*, *optional*) :
#             The hyperparameters for the optimization algorithm with keyword arguments if desired.
#
#             To set a hyperparameter, add a keyword argument that refers to one of the hyperparameters.
#
#         Notes:
#         ----------
#         - Any hyperparameters not specified will be set to their default values.
#
#         - Hyperparameters that are specified but not used within the specified algorithm will be discarded.
#             - The user will receive a warning when this occurs.
#
#         - The internal memory will automatically be initialized when Optimizers is instantiated.
#
#         Example:
#         -----
#         >>> from gardenpy.utils.algorithms import Optimizers
#         >>> optim = Optimizers('adam', hyperparameters={'alpha': 1e-2, 'lambda_d': 1e-3, 'ams': True})
#         """
#         # optimization algorithms
#         self.algorithms = [
#             'adam',
#             'sgd',
#             'rmsp'
#         ]
#
#         # internal optimization algorithm parameters
#         self._algorithm = self._check_algorithm(algorithm)
#         self._hyps = self._get_hyperparams(hyperparameters, kwargs)
#
#         # optimization algorithm
#         self._optim = self._get_optim()
#
#         # internal memory
#         self._correlator = correlator
#         if correlator:
#             # instance memorization
#             self._tensors = []
#             self._full = []
#             self._memory = None
#         else:
#             # set memorization
#             self._memory = self._get_memory()
#
#     def _check_algorithm(self, algorithm):
#         # checks whether the optimization algorithm is valid
#         if algorithm in self.algorithms:
#             # valid optimization algorithm
#             return algorithm
#         else:
#             # invalid optimization algorithm
#             raise ValueError(
#                 f"Invalid optimization algorithm: '{algorithm}'\n"
#                 f"Choose from: '{[alg for alg in self.algorithms]}'"
#             )
#
#     def _get_hyperparams(self, hyperparams, kwargs):
#         # defines optimization algorithm hyperparameters
#         # default optimization algorithm hyperparameters
#         default = {
#             'adam': {
#                 'alpha': 1e-3,
#                 'lambda_d': 0.0,
#                 'beta1': 0.9,
#                 'beta2': 0.999,
#                 'epsilon': 1e-8,
#                 'ams': False
#             },
#             'sgd': {
#                 'alpha': 1e-3,
#                 'lambda_d': 0.0,
#                 'mu': 0.0,
#                 'tau': 0.0,
#                 'nesterov': False
#             },
#             'rmsp': {
#                 'alpha': 1e-3,
#                 'lambda_d': 0.0,
#                 'beta2': 0.99,
#                 'mu': 0.0,
#                 'epsilon': 1e-8
#             },
#             'adag': {
#                 'alpha': 1e-2,
#                 'lambda_d': 0.0,
#                 'tau': 0.0,
#                 'nu': 0.0,
#                 'epsilon': 1e-8
#             }
#         }
#         # default optimization algorithm hyperparameter datatypes
#         dtypes = {
#             'adam': {
#                 'alpha': (float, int),
#                 'lambda_d': (float, int),
#                 'beta1': (float, int),
#                 'beta2': (float, int),
#                 'epsilon': (float, int),
#                 'ams': (bool, int)
#             },
#             'sgd': {
#                 'alpha': (float, int),
#                 'lambda_d': (float, int),
#                 'mu': (float, int),
#                 'tau': (float, int),
#                 'nesterov': (bool, int)
#             },
#             'rmsp': {
#                 'alpha': (float, int),
#                 'lambda_d': (float, int),
#                 'beta2': (float, int),
#                 'mu': (float, int),
#                 'epsilon': (float, int)
#             },
#             'adag': {
#                 'alpha': (float, int),
#                 'lambda_d': (float, int),
#                 'tau': (float, int),
#                 'nu': (float, int),
#                 'epsilon': (float, int)
#             }
#         }
#         # default optimization algorithm hyperparameter value types
#         vtypes = {
#             'adam': {
#                 'alpha': lambda x: True,
#                 'lambda_d': lambda x: 0.0 <= x < 1.0,
#                 'beta1': lambda x: 0.0 <= x < 1.0,
#                 'beta2': lambda x: 0.0 <= x < 1.0,
#                 'epsilon': lambda x: 0.0 < x <= 1e-2,
#                 'ams': lambda x: True
#             },
#             'sgd': {
#                 'alpha': lambda x: True,
#                 'lambda_d': lambda x: 0.0 <= x < 1.0,
#                 'mu': lambda x: 0.0 <= x < 1.0,
#                 'tau': lambda x: 0.0 <= x < 1.0,
#                 'nesterov': lambda x: True
#             },
#             'rmsp': {
#                 'alpha': lambda x: True,
#                 'lambda_d': lambda x: 0.0 <= x < 1.0,
#                 'beta2': lambda x: 0.0 <= x < 1.0,
#                 'mu': lambda x: 0.0 <= x < 1.0,
#                 'epsilon': lambda x: 0.0 < x <= 1e-2
#             },
#             'adag': {
#                 'alpha': lambda x: True,
#                 'lambda_d': lambda x: 0.0 <= x < 1.0,
#                 'tau': lambda x: 0.0 <= x <= 1.0,
#                 'nu': lambda x: 0.0 <= x <= 1.0,
#                 'epsilon': lambda x: 0.0 < x <= 1e-2
#             }
#         }
#         # default optimization algorithm hyperparameter conversion types
#         ctypes = {
#             'adam': {
#                 'alpha': lambda x: float(x),
#                 'lambda_d': lambda x: float(x),
#                 'beta1': lambda x: float(x),
#                 'beta2': lambda x: float(x),
#                 'epsilon': lambda x: float(x),
#                 'ams': lambda x: bool(x)
#             },
#             'sgd': {
#                 'alpha': lambda x: float(x),
#                 'lambda_d': lambda x: float(x),
#                 'mu': lambda x: float(x),
#                 'tau': lambda x: float(x),
#                 'nesterov': lambda x: bool(x)
#             },
#             'rmsp': {
#                 'alpha': lambda x: float(x),
#                 'lambda_d': lambda x: float(x),
#                 'beta2': lambda x: float(x),
#                 'mu': lambda x: float(x),
#                 'epsilon': lambda x: float(x)
#             },
#             'adag': {
#                 'alpha': lambda x: float(x),
#                 'lambda_d': lambda x: float(x),
#                 'tau': lambda x: float(x),
#                 'nu': lambda x: float(x),
#                 'epsilon': lambda x: float(x)
#             }
#         }
#
#         # instantiate hyperparameters dictionary
#         hyps = default[self._algorithm]
#
#         # combine keyword arguments and hyperparameters
#         if hyperparams and kwargs:
#             hyperparams.update(kwargs)
#         elif kwargs:
#             hyperparams = kwargs
#
#         if hyperparams and (hyps is not None) and isinstance(hyps, dict):
#             # set defined hyperparameters
#             for hyp in hyperparams:
#                 if hyp not in hyps:
#                     # invalid hyperparameter
#                     print()
#                     warnings.warn(
#                         f"\nInvalid hyperparameter for '{self._algorithm}': '{hyp}'\n"
#                         f"Choose from: '{[hyp for hyp in hyps]}'",
#                         UserWarning
#                     )
#                 elif hyp in hyps and not isinstance(hyperparams[hyp], dtypes[self._algorithm][hyp]):
#                     # invalid datatype for hyperparameter
#                     raise TypeError(
#                         f"Invalid datatype for '{self._algorithm}': '{hyp}'\n"
#                         f"Choose from: '{dtypes[self._algorithm][hyp]}'"
#                     )
#                 elif hyp in hyps and not (vtypes[self._algorithm][hyp](hyperparams[hyp])):
#                     # invalid value for hyperparameter
#                     raise TypeError(
#                         f"Invalid value for '{self._algorithm}': '{hyp}'\n"
#                         f"Conditional: '{vtypes[self._algorithm][hyp]}'"
#                     )
#                 else:
#                     # valid hyperparameter
#                     hyps[hyp] = ctypes[self._algorithm][hyp](hyperparams[hyp])
#         elif hyperparams and isinstance(hyperparams, dict):
#             # hyperparameters not taken
#             print()
#             warnings.warn(f"\n'{self._algorithm}' does not take hyperparameters", UserWarning)
#         elif hyperparams:
#             # invalid data type
#             raise TypeError(
#                 f"'hyperparameters' is not a dictionary: '{hyperparams}'\n"
#                 f"Choose from: '{[hyp for hyp in hyps]}'"
#             )
#
#         # return hyperparameters
#         return hyps
#
#     def _get_memory(self):
#         # instantiates memory dictionary
#         # required memory components for each optimization algorithm
#         memories = {
#             'adam': {
#                 'deltas_p': None,
#                 'upsilons_p': None,
#                 'iota': 1.0,
#                 'upsilons_hat_mx': None
#             },
#             'sgd': {
#                 'deltas_p': None
#             },
#             'rmsp': {
#                 'deltas_p': None,
#                 'upsilons_p': None
#             },
#             'adag': {
#                 'iota': 1.0,
#                 'tau': None
#             }
#         }
#         # return memory dictionary
#         return memories[self._algorithm]
#
#     def _get_optim(self):
#         # defines optimization algorithm
#         def adam(thetas, nablas):
#             # Adaptive Moment Estimation optimization algorithm
#             if self._hyps['lambda_d']:
#                 # weight decay
#                 nablas = nablas + self._hyps['lambda_d'] * thetas
#
#             if self._memory['deltas_p'] is not None:
#                 # zero-biased first moment estimate
#                 deltas = self._hyps['beta1'] * self._memory['deltas_p'] + (1.0 - self._hyps['beta1']) * nablas
#             else:
#                 # zero-biased initialized first moment estimate
#                 deltas = (1.0 - self._hyps['beta1']) * nablas
#             # zero-biased first moment memory update
#             self._memory['deltas_p'] = deltas
#
#             if self._memory['upsilons_p'] is not None:
#                 # zero-biased second raw moment estimate
#                 upsilons = self._hyps['beta2'] * self._memory['upsilons_p'] + (
#                             1.0 - self._hyps['beta2']) * nablas ** 2.0
#             else:
#                 # zero-biased initialized second raw moment estimate
#                 upsilons = (1.0 - self._hyps['beta2']) * nablas ** 2.0
#             # zero-biased second raw moment memory update
#             self._memory['upsilons_p'] = upsilons
#
#             # zero-bias correction
#             deltas_hat = deltas / (1.0 - self._hyps['beta1'] ** self._memory['iota'])
#             upsilons_hat = upsilons / (1.0 - self._hyps['beta2'] ** self._memory['iota'])
#             # zero-bias factor update
#             self._memory['iota'] += 1.0
#
#             if self._hyps['ams']:
#                 # strongly non-convex decaying learning rate variant
#                 if self._memory['upsilons_hat_mx'] is not None:
#                     # maximum zero-biased second raw moment estimate
#                     upsilons_hat_mx = np.maximum(self._memory['upsilons_hat_mx'], upsilons_hat)
#                 else:
#                     # initial maximum zero-biased second raw moment estimate
#                     upsilons_hat_mx = np.maximum(0.0 * upsilons_hat, upsilons_hat)
#                 upsilons_hat = upsilons_hat_mx
#                 # maximum zero-biased second raw moment memory update
#                 self._memory['upsilons_hat_mx'] = upsilons_hat_mx
#
#             # optimization
#             thetas -= self._hyps['alpha'] * deltas_hat / (np.sqrt(upsilons_hat) + self._hyps['epsilon'])
#             return thetas
#
#         def sgd(thetas, nablas):
#             # Stochastic Gradient Descent optimization algorithm
#             if self._hyps['lambda_d']:
#                 # weight decay
#                 nablas = nablas + self._hyps['lambda_d'] * thetas
#
#             if self._hyps['mu'] and (self._memory['deltas_p'] is not None):
#                 # momentum and dampening
#                 deltas = self._hyps['mu'] * self._memory['deltas_p'] + (1.0 - self._hyps['tau']) * nablas
#                 # delta memory update
#                 self._memory['deltas_p'] = deltas
#             elif self._hyps['mu']:
#                 # dampening
#                 deltas = (1.0 - self._hyps['tau']) * nablas
#                 # delta memory update
#                 self._memory['deltas_p'] = deltas
#             else:
#                 # dampening
#                 deltas = (1.0 - self._hyps['tau']) * nablas
#
#             if self._hyps['nesterov'] and self._hyps['mu']:
#                 # nesterov variant
#                 deltas = nablas + self._hyps['mu'] * deltas
#
#             # optimization
#             thetas -= self._hyps['alpha'] * deltas
#             return thetas
#
#         def rmsp(thetas, nablas):
#             # Root Mean Squared Propagation optimization algorithm
#             if self._hyps['lambda_d']:
#                 # weight decay
#                 nablas = nablas + self._hyps['lambda_d'] * thetas
#
#             if self._memory['upsilons_p'] is not None:
#                 # zero-biased second raw moment estimate
#                 upsilons = self._hyps['beta2'] * self._memory['upsilons_p'] + (
#                             1.0 - self._hyps['beta2']) * nablas ** 2.0
#             else:
#                 # zero-biased initialized second raw moment estimate
#                 upsilons = (1.0 - self._hyps['beta2']) * nablas ** 2.0
#             # zero-biased second raw moment memory update
#             self._memory['upsilons_p'] = upsilons
#
#             # delta calculation
#             deltas = nablas / (np.sqrt(upsilons) + self._hyps['epsilon'])
#
#             if self._hyps['mu'] and (self._memory['deltas_p'] is not None):
#                 # momentum
#                 deltas += self._hyps['mu'] * self._memory['deltas_p']
#                 # delta memory update
#                 self._memory['deltas_p'] = deltas
#             elif self._hyps['mu']:
#                 # delta memory update
#                 self._memory['deltas_p'] = deltas
#
#             # optimization
#             thetas -= self._hyps['alpha'] * deltas
#             return thetas
#
#         def adag(thetas, nablas):
#             # todo
#             # Adaptive Gradient Algorithm optimization algorithm
#             if self._hyps['lambda_d']:
#                 # weight decay
#                 nablas = nablas + self._hyps['lambda_d'] * thetas
#
#             # learning rate adaption
#             alpha_hat = self._hyps['alpha'] / ((self._memory['iota'] - 1.0) * self._hyps['nu'] + 1.0)
#             # learning rate factor update
#             self._memory['iota'] += 1.0
#
#             if self._memory['tau'] is not None:
#                 # state-sum calculation
#                 tau = self._memory['tau'] + nablas ** 2.0
#             else:
#                 # state-sum initialization
#                 tau = 0.0 * thetas + self._hyps['tau']
#             # state-sum memory update
#             self._memory['tau'] = tau
#
#             # optimization
#             thetas -= alpha_hat * nablas / (np.sqrt(tau) + self._hyps['epsilon'])
#             return thetas
#
#         # optimization algorithm dictionary
#         optim_funcs = {
#             'adam': adam,
#             'sgd': sgd,
#             'rmsp': rmsp,
#             'adag': adag
#         }
#
#         # return optimization algorithm
#         return optim_funcs[self._algorithm]
#
#     def optimize(self, thetas: Union[Tensor, np.ndarray], nablas: Union[Tensor, np.ndarray]) -> Union[
#         Tensor, np.ndarray]:
#         r"""
#         **Optimizes the thetas of a model based on the specified gradients.**
#
#         This function ideally supports GardenPy Tensors, but is compatible with NumPy Arrays.
#
#         Parameters:
#         ----------
#         - **thetas** : (*Union[Tensor, np.ndarray]*)
#             The thetas.
#         - **nablas** : (*Union[Tensor, np.ndarray]*)
#             The gradients of the thetas.
#
#         Returns:
#         ----------
#         - **thetas** : (*Union[Tensor, np.ndarray]*)
#             The optimized thetas.
#
#         Notes:
#         ----------
#         - The optimized thetas will retain the same datatype as the initial thetas.
#         - The optimized thetas will support automatic tracking for automatic differentiation if it is a Tensor.
#
#         - The internal memory will be accessed automatically upon the function call.
#
#         Example:
#         ----------
#         >>> from gardenpy.utils.objects import Tensor
#         >>> from gardenpy.utils.algorithms import Optimizers
#         >>> optim = Optimizers('adam')
#         >>> theta = Tensor([0, 1])
#         >>> nabla = Tensor([0.2, 0.5])
#         >>> theta = optim.optimize(theta, nabla)
#         """
#         if not isinstance(thetas, (Tensor, np.ndarray)):
#             # invalid datatype
#             raise TypeError(f"'thetas' is not a Tensor or NumPy Array: '{thetas}'")
#         if not isinstance(nablas, (Tensor, np.ndarray)):
#             # invalid datatype
#             raise TypeError(f"'nablas' is not a Tensor or NumPy Array: '{nablas}'")
#
#         # return updated thetas
#         if isinstance(nablas, Tensor):
#             # nablas conversion
#             nablas = nablas.to_array()
#         if isinstance(thetas, Tensor):
#             # Tensor
#             if self._correlator:
#                 if thetas.id in self._tensors:
#                     # get memory instance
#                     self._memory = self._full[self._tensors.index(thetas.id)]
#                 else:
#                     # set memory instance
#                     self._tensors.append(thetas.id)
#                     self._full.append(self._get_memory())
#                     self._memory = self._full[self._tensors.index(thetas.id)]
#             # perform optimization
#             result = Tensor(self._optim(thetas.to_array(), nablas))
#             # maintain id
#             result.id = thetas.id
#             return result
#         else:
#             # numpy array
#             result = self._optim(thetas, nablas)
#             return result
