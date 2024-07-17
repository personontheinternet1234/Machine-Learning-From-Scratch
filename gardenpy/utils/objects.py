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
            'relations': [],
            'origin': None
        }

    def to_np(self):
        return self.tensor

    def __matmul__(self, other):
        obj = other
        if isinstance(obj, Tensor):
            obj = other.to_np()
        self._tracker['operations'].append('matmul_m')
        result = Tensor(self.tensor @ obj)
        result._tracker['origin'] = [self, other]
        self._tracker['relations'].append([other, result])
        if isinstance(other, Tensor):
            other._tracker['operations'].append('matmul_s')
            other._tracker['relations'].append([self, result])
        return result

    def __mul__(self, other):
        obj = other
        if isinstance(obj, Tensor):
            obj = other.to_np()
        self._tracker['operations'].append('mul')
        result = Tensor(self.tensor * obj)
        result._tracker['origin'] = [self, other]
        self._tracker['relations'].append([other, result])
        if isinstance(other, Tensor):
            other._tracker['operations'].append('mul')
            other._tracker['relations'].append([self, result])
        return result

    def __truediv__(self, other):
        obj = other
        if isinstance(obj, Tensor):
            obj = other.to_np()
        self._tracker['operations'].append('truediv_n')
        result = Tensor(self.tensor / obj)
        result._tracker['origin'] = [self, other]
        self._tracker['relations'].append([other, result])
        if isinstance(other, Tensor):
            other._tracker['operations'].append('truediv_d')
            other._tracker['relations'].append([self, result])
        return result

    def __add__(self, other):
        obj = other
        if isinstance(obj, Tensor):
            obj = other.to_np()
        self._tracker['operations'].append('add')
        result = Tensor(self.tensor + obj)
        result._tracker['origin'] = [self, other]
        self._tracker['relations'].append([other, result])
        if isinstance(other, Tensor):
            other._tracker['operations'].append('add')
            other._tracker['relations'].append([self, result])
        return result

    def __sub__(self, other):
        obj = other
        if isinstance(obj, Tensor):
            obj = other.to_np()
        self._tracker['operations'].append('sub_s')
        result = Tensor(self.tensor - obj)
        result._tracker['origin'] = [self, other]
        self._tracker['relations'].append([other, result])
        if isinstance(other, Tensor):
            other._tracker['operations'].append('sub_m')
            other._tracker['relations'].append([self, result])
        return result

    def __pow__(self, other):
        obj = other
        if isinstance(obj, Tensor):
            obj = other.to_np()
        self._tracker['operations'].append('pow_b')
        result = Tensor(self.tensor ** obj)
        result._tracker['origin'] = [self, other]
        self._tracker['relations'].append([other, result])
        if isinstance(other, Tensor):
            other._tracker['operations'].append('pow_p')
            other._tracker['relations'].append([self, result])
        return result

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, key):
        return self.tensor[key]

    def __str__(self):
        return str(self.tensor)

    def __repr__(self):
        if self._type == 'mat':
            return (
                f"'{id(self)}' Internals:\n"
                f"'type': {str(self._type)}\n"
                f"'value':\n{str(self.tensor)}\n"
                f"'operations': {self._tracker['operations']}\n"
                f"'path-ids': {[[f'{id(path)}' for path in pair] for pair in self._tracker['relations']]}\n"
                f"'origin': '{[f'{id(origin)}' for origin in self._tracker['origin']] if self._tracker['origin'] is not None else None}'"
            )
        elif self._type in ('grad', 'grad_chain'):
            return (
                f"'{id(self)}' Internals:\n"
                f"'type': {str(self._type)}\n"
                f"'value':\n{str(self.tensor)}\n"
                f"'operations': {self._tracker['operations']}\n"
                f"'path-ids': {[f'{id(path)}' for path in self._tracker['relations']]}\n"
                f"'origin': '{id(self._tracker['origin']) if self._tracker['origin'] is not None else None}'"
            )
        else:
            raise ValueError('no type')
