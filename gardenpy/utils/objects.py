r"""
**Objects for GardenPy.**

Attributes:
----------
**Tensor**:
    Matrix for automatic differentiation.

Notes:
----------
- Refer to GardenPy's repository or GardenPy's docs for more information.
"""

from typing import Union

import numpy as np


class Tensor:
    r"""
    **Matrix with automatic tracking for automatic differentiation**

    This matrix is similar to a NumPy Array, but tracks operations and its relationship to other variables.

    Attributes:
    ----------
    **tensor** : (*np.ndarray*)
        The Tensor value.
    **shape** : (*tuple*)
        The Tensor shape.
    **type** : (*str*)
        The Tensor type.
    **tracker** : (*dict*)
        The Tensor's internal tracker.

    Methods:
    ----------
    **__init__(obj: Union[np.ndarray, list])** :
        Instantiates the Tensor with the specified values.

    **__matmul__(other)** :
        Performs the cross product.

    **__pow__(other)** :
        Performs the hadamard power.

    **__mul__(other)** :
        Performs hadamard multiplication.

    **__truediv__(other)** :
        Performs hadamard division.

    **__add__(other)** :
        Performs addition.

    **__sub__(other)** :
        Performs subtraction.

    **to_array()** :
        NumPy Array conversion.

    Notes:
    ----------
    - Tensor automatically tracks the equations done to it and its relationship to other variables.
    - Use 'nablas' or 'chain' from gardenpy.utils.operators to automatically differentiate.

    - Refer to GardenPy's repository or GardenPy's docs for more information.
    """
    def __init__(self, obj: Union[np.ndarray, list]):
        r"""
        **Tensor initialization.**

        Parameters:
        ----------
        **obj** : (*Union[np.ndarray, list]*)
            The object that will become a Tensor.

        Notes:
        ----------
        - Tensors act similarly to NumPy Arrays.
        - Most GardenPy functions support Tensors.

        Example:
        -----
        >>> from gardenpy.utils.objects import Tensor
        >>> tens = Tensor(np.random.randn(5, 5))
        """
        # internal array
        self.tensor = obj
        dtypes = (np.ndarray, list)
        if not isinstance(self.tensor, dtypes):
            # invalid datatype
            raise TypeError(
                f"Invalid datatype for 'obj': '{obj}'\n"
                f"Choose from: '{dtypes}'"
            )
        if not isinstance(self.tensor, np.ndarray):
            # numpy array conversion
            self.tensor = np.array(self.tensor)

        # internals
        self.shape = self.tensor.shape
        self.type = 'mat'
        self.tracker = {
            'opr': [],
            'drv': [],
            'chn': [],
            'rlt': [],
            'org': [None]
        }

    def __str__(self):
        # string conversion
        return str(self.tensor)

    def __len__(self):
        # length
        return len(self.tensor)

    def __getitem__(self, key):
        # item returner
        return self.tensor[key]

    def __matmul__(self, other):
        # matrix multiplication
        arr = other
        if isinstance(arr, Tensor):
            # convert other to an array to reduce computation
            arr = other.to_array()
        # track operation
        self.tracker['opr'].append('matmul_l')
        self.tracker['drv'].append(self._d_matmul_l)
        self.tracker['chn'].append(self._d_matmul_l_chn)
        # perform operation
        result = Tensor(self.tensor @ arr)
        # track origin
        result.tracker['org'] = [self, other]
        # track relation
        self.tracker['rlt'].append([other, result])
        if isinstance(other, Tensor):
            # track operation in secondary tensor
            other.tracker['opr'].append('matmul_r')
            other.tracker['drv'].append(self._d_matmul_r)
            other.tracker['chn'].append(self._d_matmul_r_chn)
            # track relation in secondary tensor
            other.tracker['rlt'].append([self, result])
        # return completed operation
        return result

    @staticmethod
    def _d_matmul_l(left_matrix, right_matrix):
        # matrix multiplication left derivative
        res = np.reshape(right_matrix.T, (right_matrix.shape[1], *left_matrix.shape))
        # print(res)
        # print(right_matrix.T)
        # print(res)
        # print(left_matrix)
        return res

    @staticmethod
    def _d_matmul_r(right_matrix, left_matrix):
        # matrix multiplication right derivative
        return left_matrix.T * (0.0 * right_matrix + 1.0)

    @staticmethod
    def _d_matmul_l_chn(downstream, upstream, reduce=True):
        if reduce:
            # print()
            # print(downstream)
            # print()
            # print('up')
            # print(upstream)
            # print()
            # print()
            # print()
            # print('down')
            # print(np.sum(downstream, axis=1))
            # print()
            # print(np.sum(upstream, axis=1))
            # print(downstream)
            # print(upstream)
            # print(upstream)
            # print(downstream)
            # print(np.sum(upstream * downstream.T, axis=1))
            # print('a')
            # print(upstream[0] * downstream.squeeze()[0])
            # print(upstream[1] * downstream.squeeze()[1])
            # print(downstream.shape)
            # print(upstream.shape)
            # print(downstream.T)
            # print(upstream)
            shp = upstream.shape
            res = np.reshape(downstream.T * upstream.squeeze(), shp).T
            return res
            # return np.sum(upstream * downstream.T, axis=1)
        else:
            raise NotImplementedError('not in yet')

    @staticmethod
    def _d_matmul_r_chn(downstream, upstream, reduce=True):
        if reduce:
            return downstream * upstream
        else:
            raise NotImplementedError('not in yet')

    def __pow__(self, other):
        # hadamard power
        arr = other
        if isinstance(arr, Tensor):
            # convert other to an array to reduce computation
            arr = other.to_array()
        # track operation
        self.tracker['opr'].append('pow_b')
        self.tracker['drv'].append(self._d_pow_b)
        self.tracker['chn'].append(self._d_pow_b_chn)
        # perform operation
        result = Tensor(self.tensor ** arr)
        # track origin
        result.tracker['org'] = [self, other]
        # track relation
        self.tracker['rlt'].append([other, result])
        if isinstance(other, Tensor):
            # track operation in secondary tensor
            other.tracker['opr'].append('pow_e')
            other.tracker['drv'].append(self._d_pow_e)
            other.tracker['chn'].append(self._d_pow_e_chn)
            # track relation in secondary tensor
            other.tracker['rlt'].append([self, result])
        # return completed operation
        return result

    @staticmethod
    def _d_pow_b(base, exponent):
        # hadamard power base derivative
        return exponent * (base ** (exponent - 1))

    @staticmethod
    def _d_pow_e(exponent, base):
        # hadamard power exponent derivative
        return np.log(base) * (base ** exponent)

    @staticmethod
    def _d_pow_b_chn(downstream, upstream, _=None):
        return downstream * upstream

    @staticmethod
    def _d_pow_e_chn(downstream, upstream, _=None):
        return downstream * upstream

    def __mul__(self, other):
        # hadamard multiplication
        arr = other
        if isinstance(arr, Tensor):
            # convert other to an array to reduce computation
            arr = other.to_array()
        # track operation
        self.tracker['opr'].append('mul')
        self.tracker['drv'].append(self._d_mul)
        self.tracker['chn'].append(self._d_mul_chn)
        # perform operation
        result = Tensor(self.tensor * arr)
        # track origin
        result.tracker['org'] = [self, other]
        # track relation
        self.tracker['rlt'].append([other, result])
        if isinstance(other, Tensor):
            # track operation in secondary tensor
            other.tracker['opr'].append('mul')
            other.tracker['drv'].append(self._d_mul)
            other.tracker['chn'].append(self._d_mul_chn)
            # track relation in secondary tensor
            other.tracker['rlt'].append([self, result])
        # return completed operation
        return result

    @staticmethod
    def _d_mul(multiplicand, multiplier, _=None):
        # hadamard multiplication derivative
        return multiplier * (multiplicand * 0.0 + 1.0)

    @staticmethod
    def _d_mul_chn(downstream, upstream, _=None):
        return downstream * upstream

    def __truediv__(self, other):
        # hadamard division
        arr = other
        if isinstance(arr, Tensor):
            # convert other to an array to reduce computation
            arr = other.to_array()
        # track operation
        self.tracker['opr'].append('truediv_n')
        self.tracker['drv'].append(self._d_truediv_n)
        self.tracker['chn'].append(self._d_truediv_n_chn)
        # perform operation
        result = Tensor(self.tensor / arr)
        # track origin
        result.tracker['org'] = [self, other]
        # track relation
        self.tracker['rlt'].append([other, result])
        if isinstance(other, Tensor):
            # track operation in secondary tensor
            other.tracker['opr'].append('truediv_d')
            other.tracker['drv'].append(self._d_truediv_d)
            other.tracker['chn'].append(self._d_truediv_d_chn)
            # track relation in secondary tensor
            other.tracker['rlt'].append([self, result])
        # return completed operation
        return result

    @staticmethod
    def _d_truediv_n(numerator, denominator):
        # hadamard division numerator derivative
        return denominator * (numerator * 0.0 + 1.0)

    @staticmethod
    def _d_truediv_d(denominator, numerator):
        # hadamard division denominator derivative
        return (denominator * 0.0 + 1.0) / numerator

    @staticmethod
    def _d_truediv_n_chn(downstream, upstream, _=None):
        return downstream * upstream

    @staticmethod
    def _d_truediv_d_chn(downstream, upstream, _=None):
        return downstream * upstream

    def __add__(self, other):
        # addition
        arr = other
        if isinstance(arr, Tensor):
            # convert other to an array to reduce computation
            arr = other.to_array()
        # track operation
        self.tracker['opr'].append('add')
        self.tracker['drv'].append(self._d_add)
        self.tracker['chn'].append(self._d_add_chn)
        # perform operation
        result = Tensor(self.tensor + arr)
        # track origin
        result.tracker['org'] = [self, other]
        # track relation
        self.tracker['rlt'].append([other, result])
        if isinstance(other, Tensor):
            # track operation in secondary tensor
            other.tracker['opr'].append('add')
            other.tracker['drv'].append(self._d_add)
            other.tracker['chn'].append(self._d_add_chn)
            # track relation in secondary tensor
            other.tracker['rlt'].append([self, result])
        # return completed operation
        return result

    @staticmethod
    def _d_add(addend, _=None):
        # addition derivative
        return addend * 0.0 + 1.0

    @staticmethod
    def _d_add_chn(downstream, upstream, _=None):
        return downstream * upstream

    def __sub__(self, other):
        # subtraction
        arr = other
        if isinstance(arr, Tensor):
            # convert other to an array to reduce computation
            arr = other.to_array()
        # track operation
        self.tracker['opr'].append('sub_m')
        self.tracker['drv'].append(self._d_sub_m)
        self.tracker['chn'].append(self._d_sub_m_chn)
        # perform operation
        result = Tensor(self.tensor - arr)
        # track origin
        result.tracker['org'] = [self, other]
        # track relation
        self.tracker['rlt'].append([other, result])
        if isinstance(other, Tensor):
            # track operation in secondary tensor
            other.tracker['opr'].append('sub_s')
            other.tracker['drv'].append(self._d_sub_s)
            other.tracker['chn'].append(self._d_sub_s_chn)
            # track relation in secondary tensor
            other.tracker['rlt'].append([self, result])
        # return completed operation
        return result

    @staticmethod
    def _d_sub_m(minuend, _=None):
        # subtraction minuend subtrahend derivative
        return minuend * 0.0 + 1.0

    @staticmethod
    def _d_sub_s(subtrahend, _=None):
        # subtraction subtrahend derivative
        return subtrahend * 0.0 - 1.0

    @staticmethod
    def _d_sub_m_chn(downstream, upstream, _=None):
        return downstream * upstream

    @staticmethod
    def _d_sub_s_chn(downstream, upstream, _=None):
        return downstream * upstream

    def reduce(self):
        self.tensor = np.sum(self.tensor, axis=2).T
        self.shape = self.tensor.shape

    def to_array(self) -> np.ndarray:
        r"""
        **Tensor to NumPy Array conversion**

        Parameters:
        ----------
        None.

        Returns:
        ----------
        - **array** : (*np.ndarray*)
            The NumPy Array of the Values.

        Notes:
        ----------
        - Just the value of the Tensor will be returned.
        """
        # return array
        return self.tensor

    def __repr__(self):
        # representation
        return str(self)

    def tracker_repr(self):
        # representation of the objects internals
        types = ['mat', 'grad']
        if self.type == 'mat':
            # matrix
            return (
                f"'{id(self)}' Internals:\n"
                f"'type': {str(self.type)}\n"
                f"'value':\n{str(self.tensor)}\n"
                f"'operations': {self.tracker['opr']}\n"
                f"'operation derivatives': {self.tracker['drv']}\n"
                f"'operation chain-rules': {self.tracker['chn']}\n"
                f"'path-ids': {[[f'{id(path)}' for path in pair] for pair in self.tracker['rlt']]}\n"
                f"'origin': {[f'{id(origin)}' for origin in self.tracker['org']] if self.tracker['org'] != [None] else None}\n"
            )
        elif self.type == 'grad':
            # gradient
            return (
                f"{id(self)} Internals:\n"
                f"'type': {str(self.type)}\n"
                f"'value':\n{str(self.tensor)}\n"
                f"'operations': {self.tracker['opr']}\n"
                f"'operation derivatives': {self.tracker['drv']}\n"
                f"'operation chain-rules': {self.tracker['chn']}\n"
                f"'path-ids': {[id(path) for path in self.tracker['rlt']]}\n"
                f"'origin': {id(self.tracker['org'])}\n"
            )
        else:
            # invalid type
            raise ValueError(
                f"Invalid Tensor type: '{self.tensor}'\n"
                f"Tensor must be: '{types}'"
            )
