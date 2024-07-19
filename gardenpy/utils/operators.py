r"""
'operators' includes the operators for GardenPy.

'operators' includes:
    'nabla': Automatic gradient calculation for Tensors.
    'chain': Automatic chain-ruling for the gradients of Tensors.

Refer to 'todo' for in-depth documentation on these operators.
"""

import numpy as np

from .objects import Tensor


def nabla(gradient: Tensor, respect: Tensor) -> Tensor:
    r"""
    'nabla' is a function that finds the gradient of 'gradient' with respect to 'respect' with automatic chain-ruling.

    Arguments:
        gradient: The item the gradient is with.
        respect: The item the gradient is with respect to.

    Returns:
        A Tensor of the gradient of the two Tensors.
    """
    if not isinstance(gradient, Tensor):
        # invalid datatype
        raise TypeError(f"'gradient' is not a Tensor: '{gradient}'")
    if not isinstance(respect, Tensor):
        # invalid datatype
        raise TypeError(f"'respect' is not a Tensor: '{respect}'")
    if gradient.type != 'mat':
        # invalid tensor type
        raise TypeError(f"'gradient' is not a Matrix: '{gradient}'")
    if respect.type != 'mat':
        # invalid tensor type
        raise TypeError(f"'respect' is not a Matrix: '{respect}'")

    # instantiate gradient relation
    relation = None

    # find gradient relation
    def _relate(item, target, path=None):
        nonlocal relation
        if path is None:
            # start path
            path = []
        if relation is None and isinstance(item, Tensor):
            # get downstream relations
            origins = item.tracker['org']
            path.append(item)
            if target in origins:
                # found gradient relation
                path.append(target)
                relation = path
            else:
                # search for relation
                [_relate(origin, target, path) for origin in origins]

    # check for valid relation
    _relate(gradient, respect)
    if not isinstance(relation, list):
        # no relation
        raise TypeError(
            f"No relation found between 'gradient' and 'respect'.\n"
            f"'gradient': {gradient}\n"
            f"'respect': {respect}"
        )

    def _derive(downstream, upstream):
        # find local gradients
        # find relations
        strm_result = [rlt[1] for rlt in upstream.tracker['rlt']]
        strm_other = [rlt[0] for rlt in upstream.tracker['rlt']]

        # local gradient calculations
        def _d_matmul_l(left_matrix, right_matrix):
            # matrix multiplication left derivative
            # todo: check this math
            return right_matrix @ left_matrix.T

        def _d_matmul_r(right_matrix, left_matrix):
            # matrix multiplication right derivative
            # todo: check this math
            return left_matrix.T @ right_matrix

        def _d_pow_b(base, exponent):
            # hadamard power base derivative
            return exponent * (base ** (exponent - 1))

        def _d_pow_e(exponent, base):
            # hadamard power exponent derivative
            return np.log(base) * (base ** exponent)

        def _d_mul(multiplicand, multiplier):
            # hadamard multiplication derivative
            return multiplier * (multiplicand * 0.0 + 1.0)

        def _d_truediv_n(numerator, denominator):
            # hadamard division numerator derivative
            return denominator * (numerator * 0.0 + 1.0)

        def _d_truediv_d(denominator, numerator):
            # hadamard division denominator derivative
            return numerator ** -1 * (denominator * 0.0 + 1.0)

        def _d_add(addend, _):
            # addition derivative
            return addend * 0.0 + 1.0

        def _d_sub_m(minuend, _):
            # subtraction minuend subtrahend derivative
            return minuend * 0.0 + 1.0

        def _d_sub_s(subtrahend, _):
            # subtraction subtrahend derivative
            return subtrahend * 0.0 - 1.0

        # gradient calculation correspondences
        gradients = {
            'matmul_l': _d_matmul_l,
            'matmul_r': _d_matmul_r,
            'pow_b': _d_pow_b,
            'pow_e': _d_pow_e,
            'mul': _d_mul,
            'truediv_n': _d_truediv_n,
            'truediv_d': _d_truediv_d,
            'add': _d_add,
            'sub_m': _d_sub_m,
            'sub_s': _d_sub_s
        }

        # get operation type
        operator = upstream.tracker['opr'][strm_result.index(downstream)]
        other = strm_other[strm_result.index(downstream)]
        if isinstance(other, Tensor):
            # numpy array conversion
            other = other.to_array()
        # calculate local gradient
        grad = Tensor(gradients[operator](upstream.to_array(), other))
        # set local gradient internals
        grad.type = 'grad'
        grad.tracker['opr'].append(f'd_{operator}')
        grad.tracker['rlt'] += [downstream, upstream]
        grad.tracker['org'] = downstream
        # return local gradient
        return grad

    # find first gradient
    result = _derive(relation[-2], relation[-1])
    del relation[-1]
    while 1 < len(relation):
        # chain rule local and final gradients
        result = chain(_derive(relation[-2], relation[-1]), result)
        del relation[-1]

    # return final gradient
    return result


def chain(downstream: Tensor, upstream: Tensor) -> Tensor:
    r"""
    'chain' is a function that manually chain-rules two gradients.

    Arguments:
        downstream: The downstream gradient.
        upstream: The upstream gradient.

    Returns:
        A Tensor of the chain-ruled gradient.
    """
    if not isinstance(upstream, Tensor):
        # invalid datatype
        raise TypeError(f"'upstream' is not a Tensor: '{upstream}'")
    if not isinstance(downstream, Tensor):
        # invalid datatype
        raise TypeError(f"'downstream' is not a Tensor: '{downstream}'")
    if not upstream.type == 'grad':
        # invalid tensor type
        raise TypeError(f"'upstream' is not a Gradient: '{upstream}'")
    if not downstream.type == 'grad':
        # invalid tensor type
        raise TypeError(f"'downstream' is not a Tensor: '{downstream}'")

    # check for valid relation
    down_relation = downstream.tracker['rlt'][-1]
    up_relation = upstream.tracker['org']
    if down_relation == up_relation:
        # valid relation
        # chain-rule gradients
        # todo: check this math
        result = Tensor(np.dot(upstream.to_array(), downstream.to_array()))
        # set gradient internals
        result.type = 'grad'
        result.tracker['rlt'] = downstream.tracker['rlt'] + upstream.tracker['rlt'][1:]
        result.tracker['opr'] = downstream.tracker['opr'] + upstream.tracker['opr']
        result.tracker['org'] = downstream.tracker['org']
        # return final gradient
        return result
    else:
        # no relation
        raise TypeError(
            f"No relation found between 'downstream' and 'upstream'.\n"
            f"'downstream': {downstream}\n"
            f"'upstream': {upstream}"
        )
