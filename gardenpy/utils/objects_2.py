import numpy as np


class Tensor2:
    _instances = []
    _idx = 0

    def __init__(self, obj: np.ndarray):
        if not isinstance(obj, np.ndarray):
            raise TypeError("'obj' must be a NumPy array")

        self._instances.append(self)
        self._id = self._idx
        self._idx += 1

        self._tensor = obj
        self._type = 'mat'
        self._tracker = {
            'opr': [],
            'drv': [],
            'chn': [],
            'rlt': [],
            'org': [None]
        }

    @property
    def id(self):
        return hex(self._id)

    @property
    def type(self):
        return self._type

    @property
    def array(self):
        return self._tensor

    @classmethod
    def reset_tracker(cls):
        for instance in cls._instances:
            if instance.type == 'mat':
                instance._tracker = {
                    'opr': [],
                    'drv': [],
                    'chn': [],
                    'rlt': [],
                    'org': [None]
                }

    @classmethod
    def reset(cls):
        cls._instances = []
        cls._idx = 0

    class PairedTensorMethod:
        def __init__(self, cls):
            self._cls = cls

        def __call__(self, **kwargs):
            main = kwargs['main']
            if isinstance(kwargs['other'], Tensor2):
                other, arr = [kwargs['other'], kwargs['other'].array]
            else:
                other, arr = 2 * [kwargs['other']]
            main._tracker['opr'].append()

    class LoneTensorMethod:
        def __init__(self, func):
            self._func = func

        def __call__(self, *args, **kwargs):



    def track_method(self, other, mth):
        arr = other
        if isinstance(arr, Tensor2):
            arr = other.array
        self._tracker['opr'].append(mth['opr_m'])
        self._tracker['drv'].append(mth['drv_m'])
        self._tracker['chn'].append(mth['chn_m'])
        result = Tensor2(self._tensor ** arr)
        result._tracker['org'] = [self, other]
        self._tracker['rlt'].append([other, result])
        if isinstance(other, Tensor2):
            other._tracker['opr'].append(mth['opr_o'])
            other._tracker['drv'].append(mth['drv_o'])
            other._tracker['chn'].append(mth['chn_o'])
            other._tracker['rlt'].append([self, result])
        return result
