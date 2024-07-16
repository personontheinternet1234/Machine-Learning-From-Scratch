import numpy as np


class Tensor:
    def __init__(self, obj):
        self.tensor = obj
        if not isinstance(self.tensor, (np.ndarray, list)):
            raise TypeError('not correct type')
        if not isinstance(self.tensor, np.ndarray):
            self.tensor = np.array(self.tensor)
        self._type = 'mat'

        self._tracker = {
            'operations': [],
            'relations': []
        }

    def to_np(self):
        return self.tensor

    def __matmul__(self, other):
        obj = other
        if isinstance(obj, Tensor):
            obj = other.to_np()
        self._tracker['operations'].append('matmul')
        result = Tensor(self.tensor @ obj)
        self._tracker['relations'].append(id(result))
        if isinstance(other, Tensor):
            other._tracker['operations'].append('matmul_m')
            other._tracker['relations'].append(id(result))
        return result

    def __mul__(self, other):
        obj = other
        if isinstance(obj, Tensor):
            obj = other.to_np()
        self._tracker['operations'].append('mul')
        result = Tensor(self.tensor * obj)
        self._tracker['relations'].append(id(result))
        if isinstance(other, Tensor):
            other._tracker['operations'].append('mul')
            other._tracker['relations'].append(id(result))
        return result

    def __truediv__(self, other):
        obj = other
        if isinstance(obj, Tensor):
            obj = other.to_np()
        self._tracker['operations'].append('truediv')
        result = Tensor(self.tensor / obj)
        self._tracker['relations'].append(id(result))
        if isinstance(other, Tensor):
            other._tracker['operations'].append('truediv_d')
            other._tracker['relations'].append(id(result))
        return result

    def __add__(self, other):
        obj = other
        if isinstance(obj, Tensor):
            obj = other.to_np()
        self._tracker['operations'].append('add')
        result = Tensor(self.tensor + obj)
        self._tracker['relations'].append(id(result))
        if isinstance(other, Tensor):
            other._tracker['operations'].append('add')
            other._tracker['relations'].append(id(result))
        return result

    def __sub__(self, other):
        obj = other
        if isinstance(obj, Tensor):
            obj = other.to_np()
        self._tracker['operations'].append('sub')
        result = Tensor(self.tensor - obj)
        self._tracker['relations'].append(id(result))
        if isinstance(other, Tensor):
            other._tracker['operations'].append('sub_m')
            other._tracker['relations'].append(id(result))
        return result

    def __str__(self):
        return f"{str(self.tensor)}"

    def __repr__(self):
        return (
            f"'{id(self.tensor)}' Internal:\n"
            f"'type': '{str(self._type)}'\n"
            f"'value':\n'{str(self.tensor)}'\n"
            f"'path': '{str(self._tracker)}'"
        )
