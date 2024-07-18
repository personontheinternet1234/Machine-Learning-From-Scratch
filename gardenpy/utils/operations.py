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
            other = other.to_array()
        result = Tensor(back[operation_type](rspc.to_array(), other))
        result._type = 'grad'
        result._tracker['operations'].append(f'd_{operation_type}')
        result._tracker['relations'] += [grad, rspc]
        result._tracker['origin'] = grad
        return result
    else:
        raise ValueError('no relation')


def test_tracker(itm1, itm2):
    tracked = False
    track_chain = [itm2]

    def collect(item, relation, chain_track=None):
        nonlocal tracked
        nonlocal track_chain
        if chain_track is None:
            chain_track = [relation]
        if isinstance(item, Tensor):
            origins = item._tracker['origin']
            if relation in origins:
                tracked = True
                track_chain = chain_track
                chain_track.append(item)
                chain_track.append(relation)
                return True
            chains = [collect(sublist, relation, chain_track) for sublist in item._tracker['origin']]
            chain_track.append(item)
            return id(item), chains, chain_track
        else:
            return False

    collect(itm1, itm2)

    final_chain = []

    def find_chain(items):
        nonlocal final_chain
        filtered = [itm for itm in items if isinstance(itm, Tensor)]
        valid = [itm for flt in filtered for itm in flt._tracker['origin'] if itm in track_chain]
        for val in valid:
            final_chain.append(val)
            find_chain(val)
    find_chain(reversed(track_chain))

    if not tracked:
        raise ValueError('no relation')
    return track_chain


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
        result = Tensor(np.dot(grad_loc.to_array(), grad_glob.to_array()))  # todo: check this math
        result._type = 'grad_chain'
        result._tracker['relations'] = grad_glob._tracker['relations'] + grad_loc._tracker['relations'][1:]
        result._tracker['operations'] = (grad_glob._tracker['operations']) + grad_loc._tracker['operations']
        result._tracker['origin'] = grad_glob._tracker['origin']
        return result
    else:
        raise TypeError('no relation')
