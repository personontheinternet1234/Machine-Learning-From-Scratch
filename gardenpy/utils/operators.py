r"""
**Operators for GardenPy.**

Attributes:
----------
**nabla**:
    Gradient calculation for Tensors.
**chain**:
    Chain ruling for Tensors.

Notes:
----------
- Refer to GardenPy's repository or GardenPy's docs for more information.
"""

from .objects import Tensor


def nabla(gradient: Tensor, respect: Tensor) -> Tensor:
    r"""
    **Gradient calculation for Tensors.**

    Calculates the gradient of 'gradient' with respect to 'respect'.

    Parameters:
    ----------
    **gradient** : (*Tensor*):
        The final value to calculate the gradient with.
    **respect** : (*Tensor*):
        The value the gradient is calculated with respect to.

    Returns:
    ----------
    - **grad** : (*Tensor*)
            The calculated gradient.

    Notes:
    ----------
    - Automatic differentation automatically chain-rules.
    - Automatic differentation relies on internal tracking from Tensors.
        - Relation detection occurs within 'nabla'.
        - You must refer to a variable you saved for the automatic detection to relate your two Tensors.
    - The two Tensors must have type 'mat'. To chain rule two gradients, use 'chain'.

    Example:
    ----------
    >>> from gardenpy.utils.objects import Tensor
    >>> from gardenpy.utils.operators import nabla
    >>> tens1 = Tensor([[5, 4, 3]])
    >>> tens2 = Tensor([[1, 2, 3]])
    >>> tens3 = tens1 + tens2
    >>> grad_tens3_tens1 = nabla(tens3, tens1)
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

        # get operation type
        operator = upstream.tracker['opr'][strm_result.index(downstream)]
        derivative = upstream.tracker['drv'][strm_result.index(downstream)]
        other = strm_other[strm_result.index(downstream)]
        if isinstance(other, Tensor):
            # numpy array conversion
            other = other.to_array()
        # calculate local gradient
        grad = Tensor(derivative(upstream.to_array(), other))
        # set local gradient internals
        grad.type = 'grad'
        grad.tracker['opr'].append(f'd_{operator}')
        grad.tracker['drv'].append(derivative)
        grad.tracker['chn'].append(upstream.tracker['chn'])
        grad.tracker['rlt'] += [downstream, upstream]
        grad.tracker['org'] = downstream
        # grad.id = downstream.id + 1
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
    **Chain ruling for Tensors.**

    Calculates the gradient that is chain-ruled from the downstream and upstream gradients.

    Parameters:
    ----------
    **downstream** : (*Tensor*):
        The downstream gradient.
    **respect** : (*Tensor*):
        The upstream gradient.

    Returns:
    ----------
    - **grad** : (*Tensor*)
            The calculated gradient.

    Notes:
    ----------
    - Automatic differentation automatically chain-rules.
        - This seperate call allows chain-ruling to increase efficieny if part of the gradient has already been calculated.
    - Automatic differentation relies on internal tracking from Tensors.
        - The two gradients must have a connecting variable for 'chain' to detect their relationship.
        - You must refer to a variable you saved for the automatic detection to relate your two Tensors.
    - The two Tensors must have type 'grad'. To chain rule two matrices, use 'nabla'.

    Example:
    ----------
    >>> from gardenpy.utils.objects import Tensor
    >>> from gardenpy.utils.operators import nabla, chain
    >>> tens1 = Tensor([[5, 4, 3]])
    >>> tens2 = Tensor([[1, 2, 3]])
    >>> tens3 = Tensor([[6, 7, 8]])
    >>> tens4 = tens1 + tens2
    >>> tens5 = tens4 + tens3
    >>> grad_tens3_tens1 = nabla(tens3, tens1)
    >>> grad_tens5_tens3 = nabla(tens5, tens3)
    >>> grad_tens5_tens1 = chain(grad_tens5_tens3, grad_tens3_tens1)
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
        result = Tensor(upstream.tracker['chn'][0][0](downstream.to_array(), upstream.to_array(), downstream.tracker['org']))
        # set gradient internals
        result.type = 'grad'
        result.tracker['rlt'] = downstream.tracker['rlt'] + upstream.tracker['rlt'][1:]
        result.tracker['opr'] = downstream.tracker['opr'] + upstream.tracker['opr']
        result.tracker['drv'] = downstream.tracker['drv'] + upstream.tracker['drv']
        result.tracker['chn'] = downstream.tracker['chn'] + upstream.tracker['chn']
        result.tracker['org'] = downstream.tracker['org']
        # result.id = downstream.id + 1

        # return final gradient
        return result
    else:
        # no relation
        raise TypeError(
            f"No relation found between 'downstream' and 'upstream'.\n"
            f"'downstream': {downstream}\n"
            f"'upstream': {upstream}"
        )
