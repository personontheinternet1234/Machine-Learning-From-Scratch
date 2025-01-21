# making sure that docstrings keep the same format

from typing import List, Tuple, Optional
import numpy as np


class Activators:
    r"""
    **Activation algorithms for arrays.**

    If used with GardenPy's Tensors, activation function utilizes autograd methods.
    These activation functions can be used with NumPy arrays, but won't utilize autograd.
    The derivative of these activation functions can be called if using NumPy arrays.

    Supports:
        - Softmax
        - Rectified Linear Unit (ReLU)
        - Leaky Rectified Linear Unit (Leaky ReLU)
        - Sigmoid
        - Tanh
        - Softplus
        - Mish
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
        **Set activator method and hyperparameters.**

        Any hyperparameters that remain unfilled are set to their default value.
        Supports autograd with Tensors or raw operations with NumPy arrays.

        softmax (Softmax)
            - None
        relu (Rectified Linear Unit / ReLU)
            - None
        lrelu (Leaky Rectified Linear Unit / Leaky ReLU)
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
            ikwiad (bool), default = False: Turns off all warning messages ("I know what I am doing" - ikwiad).
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
        self._set_activator()

    @classmethod
    def methods(cls) -> list:
        r"""
        **Possible activator methods.**

        Returns:
            list: Possible activator methods.
        """
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
        if method not in Activators._methods:
            raise ValueError(
                f"Attempted call to an invalid method: {method}.\n"
                f"Choose from: {Activators._methods}."
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
            # softmax
            def __init__(self):
                super().__init__(prefix='softmax')

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return np.exp(x) / np.sum(np.exp(x))

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return (np.sum(np.exp(x)) * np.exp(x) - np.exp(2.0 * x)) / (np.sum(np.exp(x)) ** 2.0)

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                # todo: correct implementation
                raise NotImplementedError("Currently not implemented.")

        class _ReLU(Tensor.LoneTensorMethod):
            # relu
            def __init__(self):
                super().__init__(prefix='relu')

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return np.maximum(0.0, x)

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return np.where(x > 0.0, 1.0, 0.0)

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                try:
                    return down @ up
                except ValueError:
                    return down * up

        class _LeakyReLU(Tensor.LoneTensorMethod):
            # leaky relu
            def __init__(self):
                super().__init__(prefix='lrelu')

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return np.maximum(h['beta'] * x, x)

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return np.where(x > 0.0, 1.0, h['beta'])

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                try:
                    return down @ up
                except ValueError:
                    return down * up

        class _Sigmoid(Tensor.LoneTensorMethod):
            # sigmoid
            def __init__(self):
                super().__init__(prefix='sigmoid')

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return (np.exp(-x) + 1.0) ** -1.0

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return np.exp(-x) / ((np.exp(-x) + 1.0) ** 2.0)

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                try:
                    return down @ up
                except ValueError:
                    return down * up

        class _Tanh(Tensor.LoneTensorMethod):
            # tanh
            def __init__(self):
                super().__init__(prefix='tanh')

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return np.tanh(x)

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return np.cosh(x) ** -2.0

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                try:
                    return down @ up
                except ValueError:
                    return down * up

        class _Softplus(Tensor.LoneTensorMethod):
            # softplus
            def __init__(self):
                super().__init__(prefix='softplus')

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return np.log(np.exp(h['beta'] * x) + 1.0) / h['beta']

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return h['beta'] * np.exp(h['beta'] * x) / (h['beta'] * np.exp(h['beta'] * x) + h['beta'])

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                try:
                    return down @ up
                except ValueError:
                    return down * up

        class _Mish(Tensor.LoneTensorMethod):
            # mish
            def __init__(self):
                super().__init__(prefix='mish')

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return x * np.tanh(np.log(np.exp(h['beta'] * x) + 1.0) / h['beta'])

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return (
                    np.tanh(np.log(np.exp(h['beta'] * x) + 1.0) / h['beta']) +
                    x * (np.cosh(np.log(np.exp(h['beta'] * x) + 1.0) / h['beta']) ** -2.0) *
                    (h['beta'] * np.exp(h['beta'] * x) / (h['beta'] * np.exp(h['beta'] * x) + h['beta']))
                )

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                try:
                    return down @ up
                except ValueError:
                    return down * up

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
        self._op = ops[self._method]()

    def __call__(self, x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        r"""
        **Forward function call.**

        Autograd is automatically applied if Tensors are used.
        Otherwise, raw operation is applied without autograd.

        Args:
            x (Tensor | np.ndarray): Inputted array.

        Returns:
            Tensor | np.ndarray: Activated array.

        Raises:
            TypeError: If an invalid object was passed for the operation.
        """
        if isinstance(x, Tensor) and x.type == 'mat':
            return self._op.main(x)
        elif isinstance(x, np.ndarray):
            return self._op.forward(x)
        else:
            raise TypeError("Attempted activation with an object that wasn't a matrix Tensor or NumPy array.")

    def derivative(self, x: np.ndarray) -> np.ndarray:
        r"""
        **Backward function call.**

        Automatically done with autograd for Tensors.
        Raw derivative operation should only be done on NumPy arrays.

        Args:
            x (np.ndarray): Inputted array.

        Returns:
            np.ndarray: Derivative activated array.

        Raises:
            TypeError: If an invalid object was passed for the operation.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("Attempted derivative activation with an object that wasn't a NumPy array.")
        return self._op.backward(x)