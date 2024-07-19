r"""
'objects' includes the objects for GardenPy.

'objects' includes:
    'Tensor': Automatic differentiation matrices.

Refer to 'todo' for in-depth documentation on these objects.
"""

import numpy as np


class Tensor:
    def __init__(self, obj: (np.ndarray, list)):
        r"""
        'Tensor' is a NumPy Array that supports automatic differentiation and tracks its operations.
        This object currently includes the operations '@' (Matrix Multiplication), '**' (Hadamard Exponentation), '*' (Hadamard Multiplication), '/' (Hadamard Division), '+' (Addition), and '-' (Subtraction).
        Additionally, this supports the operation 'to_array' (NumPy Array Conversion).
        The operation structure promotes flexibility through the easy addition of other operations.

        Arguments:
            obj: A list or numpy array that will be converted into a GardenPy Tensor.
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
        self.type = 'mat'
        self.tracker = {
            'opr': [],
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
        # perform operation
        result = Tensor(self.tensor @ arr)
        # track origin
        result.tracker['org'] = [self, other]
        # track relation
        self.tracker['rlt'].append([other, result])
        if isinstance(other, Tensor):
            # track operation in secondary tensor
            other.tracker['opr'].append('matmul_r')
            # track relation in secondary tensor
            other.tracker['rlt'].append([self, result])
        # return completed operation
        return result

    def __pow__(self, other):
        # hadamard power
        arr = other
        if isinstance(arr, Tensor):
            # convert other to an array to reduce computation
            arr = other.to_array()
        # track operation
        self.tracker['opr'].append('pow_b')
        # perform operation
        result = Tensor(self.tensor ** arr)
        # track origin
        result.tracker['org'] = [self, other]
        # track relation
        self.tracker['rlt'].append([other, result])
        if isinstance(other, Tensor):
            # track operation in secondary tensor
            other.tracker['opr'].append('pow_e')
            # track relation in secondary tensor
            other.tracker['rlt'].append([self, result])
        # return completed operation
        return result

    def __mul__(self, other):
        # hadamard multiplication
        arr = other
        if isinstance(arr, Tensor):
            # convert other to an array to reduce computation
            arr = other.to_array()
        # track operation
        self.tracker['opr'].append('mul')
        # perform operation
        result = Tensor(self.tensor * arr)
        # track origin
        result.tracker['org'] = [self, other]
        # track relation
        self.tracker['rlt'].append([other, result])
        if isinstance(other, Tensor):
            # track operation in secondary tensor
            other.tracker['opr'].append('mul')
            # track relation in secondary tensor
            other.tracker['rlt'].append([self, result])
        # return completed operation
        return result

    def __truediv__(self, other):
        # hadamard division
        arr = other
        if isinstance(arr, Tensor):
            # convert other to an array to reduce computation
            arr = other.to_array()
        # track operation
        self.tracker['opr'].append('truediv_n')
        # perform operation
        result = Tensor(self.tensor / arr)
        # track origin
        result.tracker['org'] = [self, other]
        # track relation
        self.tracker['rlt'].append([other, result])
        if isinstance(other, Tensor):
            # track operation in secondary tensor
            other.tracker['opr'].append('truediv_d')
            # track relation in secondary tensor
            other.tracker['rlt'].append([self, result])
        # return completed operation
        return result

    def __add__(self, other):
        # addition
        arr = other
        if isinstance(arr, Tensor):
            # convert other to an array to reduce computation
            arr = other.to_array()
        # track operation
        self.tracker['opr'].append('add')
        # perform operation
        result = Tensor(self.tensor + arr)
        # track origin
        result.tracker['org'] = [self, other]
        # track relation
        self.tracker['rlt'].append([other, result])
        if isinstance(other, Tensor):
            # track operation in secondary tensor
            other.tracker['opr'].append('add')
            # track relation in secondary tensor
            other.tracker['rlt'].append([self, result])
        # return completed operation
        return result

    def __sub__(self, other):
        # subtraction
        arr = other
        if isinstance(arr, Tensor):
            # convert other to an array to reduce computation
            arr = other.to_array()
        # track operation
        self.tracker['opr'].append('sub_m')
        # perform operation
        result = Tensor(self.tensor - arr)
        # track origin
        result.tracker['org'] = [self, other]
        # track relation
        self.tracker['rlt'].append([other, result])
        if isinstance(other, Tensor):
            # track operation in secondary tensor
            other.tracker['opr'].append('sub_s')
            # track relation in secondary tensor
            other.tracker['rlt'].append([self, result])
        # return completed operation
        return result

    def to_array(self) -> np.ndarray:
        r"""
        'to_array' is a built-in function in 'Tensors'.
        This converts a tensor to a NumPy Array.

        Arguments:
            None.

        Returns:
            A NumPy Array.
        """
        # return array
        return self.tensor

    def __repr__(self):
        # representation of the objects internals
        types = ['mat', 'grad']
        if self.type == 'mat':
            # matrix
            return (
                f"'{id(self)}' Internals:\n"
                f"'type': {str(self.type)}\n"
                f"'value':\n{str(self.tensor)}\n"
                f"'operations': {self.tracker['opr']}\n"
                f"'path-ids': {[[f'{id(path)}' for path in pair] for pair in self.tracker['rlt']]}\n"
                f"'origin': {[f'{id(origin)}' for origin in self.tracker['org']] if self.tracker['org'] != [None] else None}"
            )
        elif self.type == 'grad':
            # gradient
            return (
                f"{id(self)} Internals:\n"
                f"'type': {str(self.type)}\n"
                f"'value':\n{str(self.tensor)}\n"
                f"'operations': {self.tracker['opr']}\n"
                f"'path-ids': {[id(path) for path in self.tracker['rlt']]}\n"
                f"'origin': {id(self.tracker['org']) if self.tracker['org'] is not None else None}"
            )
        else:
            # invalid type
            raise ValueError(
                f"Invalid Tensor type: '{self.tensor}'\n"
                f"Tensor must be: '{types}'"
            )
