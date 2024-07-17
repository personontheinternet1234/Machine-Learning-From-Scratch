import gc

import numpy as np

from .objects import Tensor


def nabla(grad, rspc):
    track = [rlt[1] for rlt in rspc._tracker['relations']]
    others = [rlt[0] for rlt in rspc._tracker['relations']]
    if not isinstance(rspc, Tensor):
        raise TypeError('not tensor')
    if grad in track:
        # todo: add default chain ruling here
        def d_matmul_m(_grad, _rspc):
            ...  # todo

        def d_matmul_c(_rspc, _other):
            ...  # todo

        def d_mul(_rspc, _other):
            return _other * (_rspc * 0.0 + 1.0)

        def d_truediv_n(_rspc, _other):
            return _other * (_rspc * 0.0 + 1.0)

        def d_truediv_d(_rspc, _other):
            return _other ** -1 * (_rspc * 0.0 + 1.0)

        def d_add(_rspc, _other):
            return _rspc * 0.0 + 1.0

        def d_sub_s(_rspc, _other):
            return _rspc * 0.0 + 1.0

        def d_sub_m(_rspc, _other):
            return _rspc * 0.0 - 1.0

        back = {
            'matmul_m': d_matmul_m,
            'matmul_c': d_matmul_c,
            'mul': d_mul,
            'truediv_n': d_truediv_n,
            'truediv_d': d_truediv_d,
            'add': d_add,
            'sub_s': d_sub_s,
            'sub_m': d_sub_m
        }

        operation_type = rspc._tracker['operations'][track.index(grad)]
        other = others[track.index(grad)]
        if isinstance(other, Tensor):
            other = other.to_np()
        result = Tensor(back[operation_type](rspc.to_np(), other))
        result._type = 'grad'
        result._tracker['operations'].append(f'd_{operation_type}')
        result._tracker['relations'] += [grad, rspc]
        result._tracker['origin'] = grad
        return result
    else:
        raise ValueError('no relation')


def test_tracker(itm1, itm2):
    next_itm = itm1._tracker['origin'][0]
    # print([id(itm) for itm in next_itm])
    print(id(itm2))
    print()
    while True:
        print(id(next_itm))
        if itm2 == next_itm:
            return True
        if next_itm._tracker['origin'] is not None:
            next_itm = next_itm._tracker['origin'][0]
        else:
            return False


def chain(grad_glob, grad_loc):
    if not isinstance(grad_loc, Tensor):
        raise TypeError('loc not tensor')
    if not isinstance(grad_glob, Tensor):
        raise TypeError('glob not tensor')

    if not (grad_loc._type in ('grad', 'grad_chain') and grad_glob._type in ('grad', 'grad_chain')):
        raise TypeError('not gradients')

    glob_conn = grad_glob._tracker['relations'][-1]
    loc_conn = grad_loc._tracker['origin']
    if glob_conn == loc_conn:
        result = Tensor(np.dot(grad_loc.to_np(), grad_glob.to_np()))  # todo: check this math
        result._type = 'grad_chain'
        result._tracker['relations'] = grad_glob._tracker['relations'] + grad_loc._tracker['relations'][1:]
        result._tracker['operations'] = (grad_glob._tracker['operations']) + grad_loc._tracker['operations']
        result._tracker['origin'] = grad_glob._tracker['origin']
        return result
    else:
        raise TypeError('no relation')
