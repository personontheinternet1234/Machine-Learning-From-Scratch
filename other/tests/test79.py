# reading nabla methods to figure out how i managed to break something

@staticmethod
def nabla(grad: 'Tensor', wrt: 'Tensor', *, binary: bool = True) -> 'Tensor':
    # check tensors
    if not isinstance(grad, Tensor) or grad._type != 'mat':
        raise TypeError("'grad' must be a Tensor with the matrix type")
    if not isinstance(wrt, Tensor) or wrt._type != 'mat':
        raise TypeError("'wrt' must be a Tensor with the matrix type")

    # set gradient relation
    if binary:
        relation = None
    else:
        relation = []

    def _relate(item, target, trace=None):
        nonlocal relation
        if trace is None:
            # reset trace
            trace = []
        # NB: This only gets origins if the item is a matrix.
        # Tracing gradients through gradients is possible, but requires a lot of modification and significantly
        # increases computational time, even if it's never used.
        # Tensors allow operations on gradients for simplicity, but don't allow autograd with them.
        if binary and (relation is None) and isinstance(item, Tensor) and item._type == 'mat':
            # get origins
            origins = item._tracker['org']
            trace.append(item)
            if target in origins:
                # related
                trace.append(target)
                relation = trace.copy()
            else:
                # relation search
                [_relate(item=origin, target=target, trace=trace) for origin in origins]
        elif not binary and isinstance(item, Tensor) and item._type == 'mat':
            # get origins
            origins = item._tracker['org']
            trace.append(item)
            if target in origins:
                # related
                trace.append(target)
                relation.append(trace.copy())
            else:
                # relation search
                [_relate(item=origin, target=target, trace=trace) for origin in origins]

    # relate tensors
    _relate(wrt, grad)
    if relation is None or relation == []:
        # no relation
        raise TrackingError(grad=grad, wrt=wrt)

    def _derive(down: 'Tensor', up: 'Tensor') -> 'Tensor':
        # get relations
        strm_result = [rlt[1] for rlt in up._tracker['rlt']]
        strm_other = [rlt[0] for rlt in up._tracker['rlt']]
        # get operation
        operator = up._tracker['opr'][strm_result.index(down)]
        drv_operator = up._tracker['drv'][strm_result.index(down)]
        other = strm_other[strm_result.index(down)]

        if isinstance(other, Tensor):
            # get value
            other = other._tensor
        # calculate local gradient
        try:
            # pair derivative method
            res = drv_operator(up._tensor, other)
        except TypeError:
            # lone derivative method
            res = drv_operator(up._tensor)

        # identity conversion
        if len(res.squeeze().shape) == 1:
            res = res * np.eye(res.squeeze().shape[0])
        # tensor conversion
        res = Tensor(obj=res, _gradient_override=True)

        # set local gradient internals
        res._type = 'grad'
        res._tracker['opr'].append(f'd_{operator}')
        res._tracker['drv'].append(drv_operator)
        res._tracker['chn'].append(down._tracker['chn'])
        res._tracker['rlt'] += [down, up]
        res._tracker['org'] = down
        # return local gradient
        return res

    # linear connection override
    linear_override = False
    if binary and len(relation) != 1:
        linear_override = True
    # calculate initial gradient
    if binary:
        result = _derive(down=relation[-2], up=relation[-1])
        del relation[-1]
        while 1 < len(relation):
            # chain rule gradients
            result = Tensor.chain(down=_derive(down=relation[-2], up=relation[-1]), up=result)
            del relation[-1]
    else:
        # accumulate grads
        grads = []
        track = None
        for itm in relation:
            op_res = _derive(down=itm[-2], up=itm[-1])
            del itm[-1]
            while 1 < len(itm):
                # chain rule gradients
                op_res = Tensor.chain(down=_derive(down=itm[-2], up=itm[-1]), up=op_res)
                del itm[-1]
            grads.append(op_res._tensor)
            track = op_res._tracker
        result = 0
        for grad in grads:
            result += grad
        result = Tensor(obj=result, _gradient_override=True)
        result._type = 'grad'
        result._tracker = track

    # return final gradient
    if linear_override:
        result._tags.append('linear override')
    return result