import numpy as np

from .objects import Tensor


def nabla(grad, rspc):
    id_grad = id(grad)
    id_rspc = id(rspc)
    if not isinstance(rspc, Tensor):
        raise TypeError('not tensor')
    if id_grad in rspc._tracker['relations']:
        # todo: add default chain ruling here
        def d_matmul(_grad, _rspc):
            ...  # todo

        def d_matmul_m(_grad, _rspc):
            ...  # todo

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

        operation_type = rspc._tracker['operations'][rspc._tracker['relations'].index(id_grad)]
        result = back[operation_type](grad, rspc)
        result._type = 'grad'
        result._tracker['operations'].append(f'd_{operation_type}')
        result._tracker['relations'].append([id_grad, id_rspc])
        return result
    else:
        raise ValueError('no relation')


def chain(grad_loc, grad_glob):
    if not isinstance(grad_loc, Tensor):
        raise TypeError('loc not tensor')
    if not isinstance(grad_glob, Tensor):
        raise TypeError('glob not tensor')

    if not (grad_loc._type == 'grad' and grad_glob._type == 'grad'):
        raise TypeError('not gradients')

    glob_conn = grad_glob._tracker['relations'][0][0]
    loc_conn = grad_loc._tracker['relations'][-1][1]
    if glob_conn == loc_conn:
        result = Tensor(np.dot(grad_loc.to_np(), grad_glob.to_np()))
        result._type = 'grad'
        result._tracker['relations'] = grad_loc._tracker['relations'] + grad_glob._tracker['relations']
        result._tracker['operations'] = grad_loc._tracker['operations'] + grad_glob._tracker['operations']
        return result
    else:
        raise TypeError('no relation')
