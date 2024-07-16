import numpy as np

from .objects import Tensor


def nabla(grad, rspc):
    id_grad = id(grad)
    if not isinstance(rspc, Tensor):
        raise TypeError('not tensor')
    if id_grad in rspc._tracker['objects']:
        def d_matmul(_grad, _rspc):
            ...  # todo

        def d_matmul_m(_grad, _rspc):
            ... # todo

        def d_mul(_grad, _rspc):
            ...  # todo

        def d_truediv(_grad, _rspc):
            ...  # todo

        def d_truediv_d(_grad, _rspc):
            ...  # todo

        def d_add(_grad, _rspc):
            return _grad * 0.0 + 1.0

        def d_sub(_grad, _rspc):
            ...  # todo

        def d_sub_m(_grad, _rspc):
            ...  # todo

        back = {
            'matmul': d_matmul,
            'matmul_m': d_matmul_m,
            'mul': d_mul,
            'truediv': d_truediv,
            'truediv_d': d_truediv_d,
            'add': d_add,
            'sub': d_sub,
            'sub_m': d_sub_m
        }

        operation_type = rspc._tracker['operations'][rspc._tracker['objects'].index(id_grad)]
        return back[operation_type](grad, rspc)
    else:
        raise ValueError('no relation')
