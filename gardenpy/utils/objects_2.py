r"""
objects.py

Includes GardenPy's objects.
Contains Tensor class.
"""

from typing import Dict, List, Optional, Union
import numpy as np


class Tensor:
    r"""
    GardenPy's Tensor class, supporting automatic gradient calculation.

    This Tensor class calculates gradients through the nabla function call, and chain rules gradients through the chain
    function call.
    Chain rules are automatically applied in a nabla function call, but computational efficiency can be achieved by
    using previously calculated gradients and manually signaling chain rules.

    Tensors support most dunder methods and necessary linear algebra operations.

    Both within the class and each instance of the class, tracking of every object occurs.
    These trackers do not automatically clear and accumulate over time.
    """
    _instances = []

    def __init__(self, obj: any):
        r"""
        Converts an object into a GardenPy Tensor.

        This function call will initially convert the object into a NumPy array.
        Tensor initialization automatically adds the instance to the running Tensor instance list and initializes the
        proper internals.

        Args:
            obj (any): Object consisting of numerical values to be converted into a GardenPy Tensor.

        Raises:
            TypeError: If the object didn't only contain numerical values.
        """
        # check object
        obj = np.array(obj)
        if not np.issubdtype(obj.dtype, np.number) or len(obj.shape) != 2:
            raise TypeError("'obj' must be a 2D matrix containing only numerical values")

        # set tensor internals
        self._id = len(Tensor._instances)
        self._tensor = obj
        self._type = 'mat'
        self._tracker: Dict[Union[str, List[any], Tensor]] = {'opr': [], 'drv': [], 'chn': [], 'rlt': [], 'org': []}

        # update instances
        Tensor._instances.append(self)

    def __repr__(self):
        return str(self._tensor)

    r"""
    Properties.
    """

    @property
    def id(self) -> str:
        r"""
        ID of the Tensor instance in hexadecimal.
        Correlates to the index within the class instance list.
        If the instance of the Tensor has been deleted, will return None.

        Returns:
            str | NoneType: Current Tensor ID.
        """
        return hex(self._id)

    @property
    def type(self) -> str:
        r"""
        Tensor type, ranging from 'mat' for matrices, 'grad' for gradients, and 'deleted' for deleted instances.

        Returns:
            str: Tensor type.
        """
        return self._type

    @property
    def tracker(self) -> dict:
        r"""
        Gets the internal Tensor tracker, referencing pointers to each object or the Tensor ID if applicable.
        The tracker keeps track of the item ID, forward operations, derivative operations, chain-rule operations, object
        relationships, and origin relationships.

        Returns:
            dict: Tensor tracker.
        """
        alt_tracker = self._tracker

        alt_tracker['rlt'] = [
            [itm[0].id if isinstance(itm[0], Tensor) else hex(id(itm[0])),
             itm[1].id if isinstance(itm[1], Tensor) else hex(id(itm[1]))]
            for itm in alt_tracker['rlt']
        ]
        alt_tracker['org'] = [
            itm.id if isinstance(itm, Tensor) else hex(id(itm))
            for itm in alt_tracker['org']
        ]

        return {'itm': self.id, **alt_tracker}

    @property
    def array(self) -> np.ndarray:
        r"""
        Tensor NumPy Value.

        Returns:
            np.ndarray: NumPy array value of Tensor.
        """
        return self._tensor

    r"""
    Instance calls.
    """

    def instance_grad_reset(self) -> None:
        # reset tensor tracker
        self._tracker: Dict[Union[str, List[any], Tensor]] = {'opr': [], 'drv': [], 'chn': [], 'rlt': [], 'org': []}
        return None

    def instance_reset(self) -> None:
        # reset tensor
        Tensor._instances[self._id] = None
        self._id = None
        self._tensor = None
        self._type = 'deleted'
        self._tracker = None
        return None

    r"""
    Class methods.
    """

    @classmethod
    def get_instances(cls) -> list:
        # return instances
        return [instance.id if instance is not None else None for instance in cls._instances]

    @classmethod
    def reset(cls) -> None:
        # reset instance
        for instance in cls._instances:
            instance.instance_reset()
        cls._instances = []
        return None

    @classmethod
    def grad_reset(cls) -> None:
        for instance in cls._instances:
            # reset tensor tracker
            instance.instance_grad_reset()
            if instance.type == 'grad':
                # reset gradients
                instance.instance_reset()
        return None

    @classmethod
    def instance_replace(cls, replaced: Union['Tensor', str, int], replacer: Union['Tensor', str, int]) -> None:
        # attempt hex to int conversion
        try:
            replaced = int(replaced, 16)
        finally:
            pass
        try:
            replacer = int(replacer, 16)
        finally:
            pass

        # get replaced tensor
        if isinstance(replaced, Tensor) and replaced.type != 'deleted':
            # use object reference
            replaced_id = replaced._id
        elif isinstance(replaced, int) and replaced < len(cls._instances):
            # use index reference
            replaced_id = replaced
            replaced = cls._instances[replaced_id]
        else:
            # invalid reference
            raise ValueError("'replaced' must reference a non-deleted Tensor")

        # get replacer tensor
        if isinstance(replacer, Tensor) and replacer.type != 'deleted':
            # use object reference
            replacer_id = replacer._id
        elif isinstance(replacer, int) and replacer < len(cls._instances):
            # use index reference
            replacer_id = replacer
            replacer = cls._instances[replacer_id]
        else:
            # invalid reference
            raise ValueError("'replacer' must reference a non-deleted Tensor")

        # reset replaced tensor
        replaced.instance_reset()
        # move replacer tensor
        replacer._id = replaced_id
        cls._instances[replaced_id] = replacer
        cls._instances[replacer_id] = None
        return None

    @classmethod
    def clear_removed(cls) -> None:
        # filter deleted tensors
        cls._instances = [instance for instance in cls._instances if instance is not None]
        for i, itm in enumerate(cls._instances):
            # update id
            itm._id = i
        return None

    @classmethod
    def zero_grad(cls) -> None:
        cls.grad_reset()
        cls.clear_removed()

    r"""
    Tensor tracking methods.
    """

    class TensorMethod:
        def __init__(self, prefix):
            self._prefix = str(prefix)

        @staticmethod
        def _update_track(obj: 'Tensor', opr: any, drv: any, chn: any, rlt: any) -> None:
            # tracker update
            obj._tracker['opr'].append(opr)
            obj._tracker['drv'].append(drv)
            obj._tracker['chn'].append(chn)
            obj._tracker['rlt'].append(rlt)
            return None

        @staticmethod
        def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            r"""
            Chain method.

            Args:
                down (np.ndarray): Downstream gradient.
                up (np.ndarray): Upstream gradient.

            Returns:
                np.ndarray: Result.
            """
            raise NotImplementedError("'chain' must be implemented in a subclass")

    class LoneTensorMethod(TensorMethod):
        @staticmethod
        def forward(main: np.ndarray) -> np.ndarray:
            r"""
            Forward function.

            Args:
                main (np.ndarray): Main array.

            Returns:
                np.ndarray: Result.
            """
            raise NotImplementedError("'forward' must be implemented in a subclass")

        @staticmethod
        def backward(main: np.ndarray) -> np.ndarray:
            r"""
            Backward function.

            Args:
                main (np.ndarray): Main array.

            Returns:
                np.ndarray: Result.
            """
            raise NotImplementedError("'backward' must be implemented in a subclass")

        def call(self, main: 'Tensor') -> 'Tensor':
            r"""
            Function call.

            Args:
                main (Tensor): Non-deleted input Tensor.

            Returns:
                Tensor: Output Tensor.
            """
            # check tensor
            if not (isinstance(main, Tensor)) or main.type == 'deleted':
                raise TypeError("'main' must be a non-deleted Tensor")
            # calculate result
            result = Tensor(self.forward(main._tensor))
            result._tracker['org'] = [main, None]
            # track main
            self._update_track(main, f"{self._prefix}", self.backward, self.chain, [None, result])
            # return result
            return result

    class PairedTensorMethod(TensorMethod):
        @staticmethod
        def forward(main: np.ndarray, other: Optional[np.ndarray]) -> np.ndarray:
            r"""
            Forward function.

            Args:
                main (np.ndarray): Main array.
                other (np.ndarray, optional): Other array.

            Returns:
                np.ndarray: Result.
            """
            raise NotImplementedError("'forward' must be implemented in a subclass")

        @staticmethod
        def backward(main: np.ndarray, other: Optional[np.ndarray]) -> np.ndarray:
            r"""
            Backward function.

            Args:
                main (np.ndarray): Main array.
                other (np.ndarray, optional): Other array.

            Returns:
                np.ndarray: Result.
            """
            raise NotImplementedError("'backward' must be implemented in a subclass")

        @staticmethod
        def backward_o(other: np.ndarray, main: np.ndarray) -> np.ndarray:
            r"""
            Other backward function.

            Args:
                main (np.ndarray): Main array.
                other (np.ndarray): Other array.

            Returns:
                np.ndarray: Result.
            """
            raise NotImplementedError("'backward_o' must be implemented in a subclass")

        @staticmethod
        def chain_o(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            r"""
            Other chain method.

            Args:
                down (np.ndarray): Downstream gradient.
                up (np.ndarray): Upstream gradient.

            Returns:
                np.ndarray: Result.
            """
            raise NotImplementedError("'chain_o' must be implemented in a subclass")

        def call(self, main: 'Tensor', other: 'Tensor') -> 'Tensor':
            r"""
            Function call.

            Args:
                main (Tensor): Non-deleted input Tensor.
                other (Tensor): Second non-deleted input Tensor.

            Returns:
                Tensor: Output Tensor.
            """
            # check tensor
            if not (isinstance(main, Tensor)) or main.type == 'deleted':
                raise TypeError("'main' must be a non-deleted Tensor")

            # set array
            if isinstance(other, Tensor):
                arr = other._tensor
            else:
                arr = other

            # calculate result
            result = Tensor(self.forward(main._tensor, arr))
            result._tracker['org'] = [main, other]
            # track main
            self._update_track(main, f"{self._prefix}", self.backward, self.chain, [other, result])
            if isinstance(other, Tensor):
                # track other
                self._update_track(other, f"{self._prefix}_o", self.backward_o, self.chain_o, [main, result])
            # return result
            return result

    r"""
    Gradient calculation methods.
    """

    @staticmethod
    def nabla(grad: 'Tensor', wrt: 'Tensor') -> 'Tensor':
        # check tensors
        if not isinstance(grad, Tensor) or grad._type != 'mat':
            raise TypeError("'grad' must be a Tensor with the matrix type")
        if not isinstance(wrt, Tensor) or wrt._type != 'mat':
            raise TypeError("'wrt' must be a Tensor with the matrix type")

        # set gradient relation
        relation = None

        def _relate(item, target, trace=None):
            nonlocal relation
            if not trace:
                # reset trace
                trace = []
            if relation is None and isinstance(item, Tensor):
                # get downstream
                origins = item._tracker['org']
                trace.append(item)
                if target in origins:
                    # related
                    trace.append(target)
                    relation = trace
                else:
                    # relation search
                    [_relate(origin, target, trace) for origin in origins]

        # relate tensors
        _relate(grad, wrt)
        if not isinstance(relation, list):
            # no relation
            raise ValueError("No relationship found between Tensors")

        def _derive(down: 'Tensor', up: 'Tensor') -> 'Tensor':
            # get relations
            strm_result = [rlt[1] for rlt in up._tracker['rlt']]
            strm_other = [rlt[0] for rlt in up._tracker['rlt']]
            # get operation
            operator = up._tracker['opr'][strm_result.index(down)]
            derivative = up._tracker['drv'][strm_result.index(down)]
            other = strm_other[strm_result.index(down)]

            if isinstance(other, Tensor):
                # get value
                other = other._tensor
            # calculate local gradient
            try:
                # pair derivative method
                res = Tensor(derivative(up._tensor, other))
            except TypeError:
                # lone derivative method
                res = Tensor(derivative(up._tensor))

            # set local gradient internals
            res._type = 'grad'
            res._tracker['opr'].append(f'd_{operator}')
            res._tracker['drv'].append(derivative)
            res._tracker['chn'].append(down._tracker['chn'])
            res._tracker['rlt'] += [down, up]
            res._tracker['org'] = down
            # return local gradient
            return res

        # calculate initial gradient
        result = _derive(relation[-2], relation[-1])
        del relation[-1]
        while 1 < len(relation):
            # chain rule gradients
            result = Tensor.chain(_derive(relation[-2], relation[-1]), result)
            del relation[-1]

        # return final gradient
        return result

    @staticmethod
    def chain(down: 'Tensor', up: 'Tensor') -> 'Tensor':
        if not isinstance(down, Tensor) or down._type != 'grad':
            raise TypeError("'down' must be a Tensor with the gradient type")
        if not isinstance(up, Tensor) or up._type != 'grad':
            raise TypeError("'up' must be a Tensor with the gradient type")

        # check relation
        down_relation = down._tracker['rlt'][-1]
        up_relation = up._tracker['org']
        if down_relation == up_relation:
            # chain gradients
            result = Tensor(up._tracker['chn'][0][0](down._tensor, up._tensor))
            # set gradient internals
            result._type = 'grad'
            result._tracker['rlt'] = down._tracker['rlt'] + up._tracker['rlt'][1:]
            result._tracker['opr'] = down._tracker['opr'] + up._tracker['opr']
            result._tracker['drv'] = down._tracker['drv'] + up._tracker['drv']
            result._tracker['chn'] = down._tracker['chn'] + up._tracker['chn']
            result._tracker['org'] = down._tracker['org']
            # return final gradient
            return result
        else:
            # no relation
            raise ValueError("No relationship found between Tensors")

    r"""
    Built-in methods.
    """

    class _MatMul(PairedTensorMethod):
        r"""Matrix multiplication built-in method."""
        # todo: fix this class
        def __init__(self):
            super().__init__(prefix="matmul")

        @staticmethod
        def forward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return main @ other

        @staticmethod
        def backward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return ...

        @staticmethod
        def backward_o(other: np.ndarray, main: np.ndarray) -> np.ndarray:
            return ...

        @staticmethod
        def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return ...

        @staticmethod
        def chain_o(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return ...

        def __call__(self, main: 'Tensor', other: 'Tensor') -> 'Tensor':
            return self.call(main, other)

    def __matmul__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        r"""Matrix multiplication of two Tensors."""
        return Tensor._MatMul()(self, other)

    class _Pow(PairedTensorMethod):
        r"""Hadamard power built-in method."""
        def __init__(self):
            super().__init__(prefix="pow")

        @staticmethod
        def forward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return main ** other

        @staticmethod
        def backward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return other * (main ** (other - 1)) * np.eye(main.shape[1])

        @staticmethod
        def backward_o(other: np.ndarray, main: np.ndarray) -> np.ndarray:
            return np.log(main) * (main ** other) * np.eye(other.shape[1])

        @staticmethod
        def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down * up

        @staticmethod
        def chain_o(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down * up

        def __call__(self, main: 'Tensor', other: 'Tensor') -> 'Tensor':
            return self.call(main, other)

    def __pow__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        r"""Hadamard power of two Tensors."""
        return Tensor._Pow()(self, other)

    class _Mul(PairedTensorMethod):
        r"""Hadamard multiplication built-in method."""
        def __init__(self):
            super().__init__(prefix="mul")

        @staticmethod
        def forward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return main * other

        @staticmethod
        def backward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return other * (main * 0.0 + 1.0) * np.eye(main.shape[1])

        @staticmethod
        def backward_o(other: np.ndarray, main: np.ndarray) -> np.ndarray:
            return main * (other * 0.0 + 1.0) * np.eye(other.shape[1])

        @staticmethod
        def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down * up

        @staticmethod
        def chain_o(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down * up

        def __call__(self, main: 'Tensor', other: 'Tensor') -> 'Tensor':
            return self.call(main, other)

    def __mul__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        r"""Hadamard multiplication of two Tensors"""
        return Tensor._Mul()(self, other)

    class _TrueDiv(PairedTensorMethod):
        r"""Hadamard division built-in method."""
        def __init__(self):
            super().__init__(prefix="truediv")

        @staticmethod
        def forward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return main / other

        @staticmethod
        def backward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return other * (main * 0.0 + 1.0) * np.eye(main.shape[1])

        @staticmethod
        def backward_o(other: np.ndarray, main: np.ndarray) -> np.ndarray:
            return (other * 0.0 + 1.0) / main * np.eye(other.shape[1])

        @staticmethod
        def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down * up

        @staticmethod
        def chain_o(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down * up

        def __call__(self, main: 'Tensor', other: 'Tensor') -> 'Tensor':
            return self.call(main, other)

    def __truediv__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        r"""Hadamard division of two Tensors."""
        return Tensor._TrueDiv()(self, other)

    class _Add(PairedTensorMethod):
        r"""Addition built-in method."""
        def __init__(self):
            super().__init__(prefix="add")

        @staticmethod
        def forward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return main + other

        @staticmethod
        def backward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return (main * 0.0 + 1.0) * np.eye(main.shape[1])

        @staticmethod
        def backward_o(other: np.ndarray, main: np.ndarray) -> np.ndarray:
            return (other * 0.0 + 1.0) * np.eye(other.shape[1])

        @staticmethod
        def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down * up

        @staticmethod
        def chain_o(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down * up

        def __call__(self, main: 'Tensor', other: 'Tensor') -> 'Tensor':
            return self.call(main, other)

    def __add__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        r"""Addition of two Tensors."""
        return Tensor._Add()(self, other)

    class _Sub(PairedTensorMethod):
        r"""Subtraction built-in method."""
        def __init__(self):
            super().__init__(prefix="sub")

        @staticmethod
        def forward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return main - other

        @staticmethod
        def backward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return (main * 0.0 + 1.0) * np.eye(main.shape[1])

        @staticmethod
        def backward_o(other: np.ndarray, main: np.ndarray) -> np.ndarray:
            return (other * 0.0 - 1.0) * np.eye(other.shape[1])

        @staticmethod
        def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down * up

        @staticmethod
        def chain_o(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down * up

        def __call__(self, main: 'Tensor', other: 'Tensor') -> 'Tensor':
            return self.call(main, other)

    def __sub__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        r"""Subtraction of two Tensors."""
        return Tensor._Sub()(self, other)
