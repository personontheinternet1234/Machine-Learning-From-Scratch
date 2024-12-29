r"""
objects.py

Includes GardenPy's objects.
Contains Tensor class.
"""

from typing import Optional
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
            obj (any):
                Object to be converted into a GardenPy Tensor.
                Must consist of only numerical values.

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
        self._tracker = {
            'opr': [],
            'drv': [],
            'chn': [],
            'rlt': [],
            'org': []
        }

        # update instances
        Tensor._instances.append(self)

    def __repr__(self):
        r"""
        Human-readable format of the tensor.

        Returns:
            str: NumPy array representation of Tensor.
        """
        return str(self._tensor)

    ####################################################################################################################
    # Instance properties.
    ####################################################################################################################

    @property
    def id(self):
        r"""
        ID of the Tensor instance in hexadecimal.
        Correlates to the index within the class instance list.
        If the instance of the Tensor has been deleted, will return None.

        Returns:
            str | NoneType: Current Tensor ID.
        """
        return hex(self._id)

    @property
    def type(self):
        r"""
        Tensor type, ranging from 'mat' for matrices, 'grad' for gradients, and 'deleted' for deleted instances.

        Returns:
            str: Tensor type.
        """
        return self._type

    @property
    def tracker(self):
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
    def array(self):
        r"""
        Tensor NumPy Value.

        Returns:
            np.ndarray: NumPy array value of Tensor.
        """
        return self._tensor

    ####################################################################################################################
    # Instance calls.
    ####################################################################################################################

    def instance_tracker_reset(self):
        # reset tensor tracker
        self._tracker = {
            'opr': [],
            'drv': [],
            'chn': [],
            'rlt': [],
            'org': []
        }
        return None

    def instance_reset(self):
        # reset tensor
        Tensor._instances[self._id] = None
        self._id = None
        self._tensor = None
        self._type = 'deleted'
        self._tracker = None
        return None

    ####################################################################################################################
    # Class methods.
    ####################################################################################################################

    @classmethod
    def get_instances(cls):
        # return instances
        return [instance.id if instance is not None else None for instance in cls._instances]

    @classmethod
    def reset(cls):
        # reset instance
        for instance in cls._instances:
            instance.instance_reset()
        cls._instances = []
        return None

    @classmethod
    def tracker_reset(cls):
        for instance in cls._instances:
            # reset tensor tracker
            instance.instance_tracker_reset()
            if instance.type == 'grad':
                # reset gradients
                instance.instance_reset()
        return None

    @classmethod
    def instance_replace(cls, replaced, replacer):
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
    def clear_removed(cls):
        # filter deleted tensors
        cls._instances = [instance for instance in cls._instances if instance is not None]
        for i, itm in enumerate(cls._instances):
            # update id
            itm._id = i
        return None

    ####################################################################################################################
    # Tensor trackers.
    ####################################################################################################################

    class TensorMethod:
        def __init__(self, prefix):
            self._prefix = prefix

        @staticmethod
        def _update_track(obj, opr, drv, chn, rlt):
            # tracker update
            assert isinstance(obj, Tensor)
            obj._tracker['opr'].append(opr)
            obj._tracker['drv'].append(drv)
            obj._tracker['chn'].append(chn)
            obj._tracker['rlt'].append(rlt)
            return None

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
        def call(self, main: "Tensor") -> "Tensor":
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
            result = Tensor(self.forward(main._tensor, None))
            result._tracker['org'] = [main, None]
            # track main
            self._update_track(main, f"{self._prefix}", self.backward, self.chain, [None, result])
            # return result
            return result

    class PairedTensorMethod(TensorMethod):
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

        def call(self, main: "Tensor", other: "Tensor") -> "Tensor":
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

    ####################################################################################################################
    # Built-in methods.
    ####################################################################################################################

    class Matmul(PairedTensorMethod):
        r"""Matrix multiplication built-in method."""
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

        def __call__(self, main: "Tensor", other: "Tensor") -> "Tensor":
            return self.call(main, other)

    class Pow(PairedTensorMethod):
        r"""Power built-in method."""
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

        def __call__(self, main: "Tensor", other: "Tensor") -> "Tensor":
            return self.call(main, other)

    class Mul(PairedTensorMethod):
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

        def __call__(self, main: "Tensor", other: "Tensor") -> "Tensor":
            return self.call(main, other)

    class Truediv(PairedTensorMethod):
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

        def __call__(self, main: "Tensor", other: "Tensor") -> "Tensor":
            return self.call(main, other)

    class Add(PairedTensorMethod):
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

        def __call__(self, main: "Tensor", other: "Tensor") -> "Tensor":
            return self.call(main, other)

    class Sub(PairedTensorMethod):
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

        def __call__(self, main: "Tensor", other: "Tensor") -> "Tensor":
            return self.call(main, other)

    ####################################################################################################################
    # Callable forward methods.
    ####################################################################################################################

    def __matmul__(self, other):
        return Tensor.Matmul()(self, other)

    def __pow__(self, other):
        return Tensor.Pow()(self, other)

    def __mul__(self, other):
        return Tensor.Mul()(self, other)

    def __truediv__(self, other):
        return Tensor.Truediv()(self, other)

    def __add__(self, other):
        return Tensor.Add()(self, other)

    def __sub__(self, other):
        return Tensor.Sub()(self, other)

    ####################################################################################################################
    # Gradient calculation methods.
    ####################################################################################################################

    ...
