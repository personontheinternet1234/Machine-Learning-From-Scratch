r"""Internal objects."""

from warnings import warn
from typing import Callable, Dict, List, Tuple, Optional, Union
import numpy as np

from utils.errors import TrackingError


class Tensor:
    r"""
    **GardenPy's Tensor, for automatic gradient calculation and tracking.**

    --------------------------------------------------------------------------------------------------------------------

    Tensors are NumPy arrays with internals that track the operations, derivatives, and chain-rule methods of the
    functions it is run through, as well as its relationship to other arrays and Tensors it is run with.
    Through this, Tensors can effectively automatically find their relationship to other Tensors.
    Through reference to the operations run on itself and other Tensors, Tensors can automatically calculate its
    gradient through a single function call, referred to as the Tensor's autograd methods.
    See :func:`Tensor.nabla` and :func:`Tensor.chain`.

    Internally, the Tensor class tracks all instances, allowing for class-wide function calls.
    A Tensor's internals can be retrieved, but most cannot be edited directly.

    The functions for calculating and chain-ruling gradients are found within this class.

    --------------------------------------------------------------------------------------------------------------------

    Note:
        As Tensors track all instances, over time, Tensor instances will accumulate, possibly reducing computational
        efficiency.
        Additionally, every instance of a method run on a Tensor is tracked.
        If a single Tensor is constantly used, its internal tracker will accumulate these instances that have a
        significant impact on the computational efficiency of the binary search tree method of relating Tensors for
        autograd methods.
        It is recommended to run the zero_grad method every so often to avoid these possible reductions in computational
        efficiency.
        See :func:`Tensor.zero_grad`.

    Note:
        Creation of Tensor methods can be done by referencing one of the Tensor method subclasses.
        These subclasses automatically take care of anything necessary for gradient tracking for autograd methods.
        As a result, direct editing of a Tensor's internals isn't necessary and is strongly discouraged.
        See :class:`Tensor.LoneTensorMethod` and :class:`Tensor.PairedTensorMethod`.
    """
    # instance reference
    _instances: List[Union['Tensor', None]] = []
    # class ikwiad reference
    _ikwiad: bool = False

    def __init__(self, obj: any, *, _gradient_override: bool = False):
        r"""
        **Creation of a GardenPy Tensor instance.**

        ----------------------------------------------------------------------------------------------------------------

        All creation instances of a Tensor will be added to the running track of all instances.
        By default, newly created Tensors will have blank internals and be of the matrix type.

        Objects fed into a Tensor instance will first attempt NumPy array conversion.
        From there, the NumPy array will be checked to ensure it follows current requirements for Tensors.
        Tensors currently support arrays of two dimensions consisting of numbers.

        ----------------------------------------------------------------------------------------------------------------

        Args:
            obj (any): Object to be turned into a Tensor.
                Object must contain only numbers and be two-dimensional.

        Raises:
            TypeError: If the object isn't a two-dimensional NumPy array consisting of only numbers.
        """
        # check object
        obj = np.array(obj)
        if (not np.issubdtype(obj.dtype, np.number) or len(obj.shape) != 2) and not _gradient_override:
            # NB: Currently, Tensors can only be two-dimensional.
            # Due to the existence of three-dimensional gradients, this lock can be overwritten, but only internally.
            raise TypeError("'obj' must be a 2D matrix containing only numerical values")

        # set tensor internals
        self._id: Union[int, None] = None
        self._tensor: Union[np.ndarray, None] = obj
        self._type: str = 'mat'
        self._tracker: Dict[Union[str, List[any], Tensor]] = {'opr': [], 'drv': [], 'chn': [], 'rlt': [], 'org': []}

        # update instances and id
        Tensor._add_instance(self)

    def __repr__(self):
        self._is_valid_tensor(itm=self)
        return str(self._tensor)

    # Tensor properties.
    # Tensor properties refer to internal properties of Tensors.
    # These properties should only be edited using one of the Tensor's methods.
    # Editing of these properties can be done externally, but is strongly discouraged.
    # To edit one of these properties, it's recommended to either use a subclass that does it for you, or create a new
    # subclass that edits the property internally and makes an instance of the subclass externally.

    @property
    def id(self) -> Union[str, None]:
        r"""
        **Hexadecimal ID of a Tensor instance.**

        ----------------------------------------------------------------------------------------------------------------

        A Tensor's id correlating to its index within the class instance list in hexadecimal.
        If the instance of the Tensor has been deleted, will return None.
        IDs are used similarly to pointers, and internally are often used as pointers, since Python doesn't explicitly
        support the use and modification of pointers.

        ----------------------------------------------------------------------------------------------------------------

        Returns:
            str | NoneType: Current Tensor ID.
                Returns None if the function is used on a deleted Tensor.

        Raises:
            UserWarning: If the function is used on a deleted Tensor.
                Turned off by toggling ikwiad.
                See :func:`Tensor.ikwiad`.
        """
        if self._is_valid_tensor(itm=self):
            return hex(self._id)
        else:
            return None

    @property
    def type(self) -> str:
        r"""
        **String of the Tensor type.**

        ----------------------------------------------------------------------------------------------------------------

        A Tensor's type, used for checking the validity of operations and internal setup.
        Ranges from 'mat' for matrices, 'grad' for gradients, and 'deleted' for deleted instances.

        ----------------------------------------------------------------------------------------------------------------

        Returns:
            str: Tensor type.
        """
        return self._type

    @property
    def tracker(self) -> Union[dict, None]:
        r"""
        **Tensor's internal tracker.**

        ----------------------------------------------------------------------------------------------------------------

        A Tensor's tracker contains all of a Tensor's relationships to everything it has been passed through.
        It's vital for the gradient and chain-rule calculations.
        Each type of Tensor has a slightly different tracker due to the nature of forward and backward methods.
        As a result, different types of dictionaries are passed through

        The tracker keeps track of a few things necessary for autograd.

        First, it keeps track of all the operations it has gone through, referenced in the tracker as 'opr'.
        These operation references are stored as strings and aren't used for any internal computation.
        Instead, they are for user convenience in debugging.

        Second, it tracks the derivative of the operations it has gone through, referenced in the tracker as
        'drv'.
        These derivative references are stored as functions and are vital for internal computation.

        Third, it tracks the chain-rule of the operations it has gone through, referenced in the tracker as
        'chn'.
        These chain-rule references are stored as functions and are vital for internal computation.

        Fourth, it tracks the relations of the operations it has gone through, referenced in the tracker as
        'rlt'.
        These relation references are stored as a Tensor or array and are returned as an ID or pointer.
        They are necessary for gradient calculation and tracking in the autograd methods.

        Fifth, it keeps track of the place the Tensor originated from, referenced in the tracker as 'org'.
        This origin reference is stored as a Tensor or array and is returned as an ID or pointer.
        It is necessary for gradient tracking in the autograd methods.

        ----------------------------------------------------------------------------------------------------------------

        Returns:
            dict | NoneType: Dictionary of a Tensor's tracker and ID.

        Raises:
            UserWarning: If the function is used on a deleted Tensor.
                Turned off by toggling ikwiad.
                See :func:`Tensor.ikwiad`.

        Note:
            This tracker temporarily turns on ikwiad to allow printing deleted gradients in the relations.
            After the tracker has been received, ikwiad is turned back to what the user set it as.
        """
        # turn on ikwiad
        user_ikwiad = Tensor._ikwiad
        Tensor._ikwiad = True
        # get tracker reference
        alt_tracker = self._tracker

        if not self._is_valid_tensor(itm=self):
            return None
        elif self._type == 'mat':
            # matrix tracker
            alt_tracker['rlt'] = [
                [itm[0].id if isinstance(itm[0], Tensor) else hex(id(itm[0])),
                 itm[1].id if isinstance(itm[1], Tensor) else hex(id(itm[1]))]
                for itm in alt_tracker['rlt']
            ]
            alt_tracker['org'] = [itm.id if isinstance(itm, Tensor) else hex(id(itm)) for itm in alt_tracker['org']]
        elif self._type == 'grad':
            # gradient tracker
            alt_tracker['rlt'] = [itm.id if isinstance(itm, Tensor) else hex(id(itm)) for itm in alt_tracker['rlt']]
            alt_tracker['org'] = alt_tracker['org'].id

        # return to user ikwiad
        Tensor._ikwiad = user_ikwiad
        # return tracker
        return {'itm': self.id, **alt_tracker}

    @property
    def array(self) -> Union[np.ndarray, None]:
        r"""
        **Tensor's internal NumPy array.**

        ----------------------------------------------------------------------------------------------------------------

        Returns:
            np.ndarray | NoneType: Tensor's internal NumPy array.

        Raises:
            UserWarning: If the function is used on a deleted Tensor.
                Turned off by toggling ikwiad.
                See :func:`Tensor.ikwiad`.
        """
        self._is_valid_tensor(itm=self)
        return self._tensor

    # External alteration of a Tensor's properties.
    # Resets or alters the internal properties of a Tensor.
    # Resetting a Tensor's properties periodically is important to prevent memory leaks.
    # Class methods utilize these Tensor methods across all instances of a Tensor.
    # These Tensor methods can be used if only one Tensor is to be edited.

    def instance_grad_reset(self) -> None:
        r"""
        **Reset a Tensor's internal tracker.**

        ----------------------------------------------------------------------------------------------------------------

        Raises:
            UserWarning: If the function is used on a deleted Tensor.
                Turned off by toggling ikwiad.
                See :func:`Tensor.ikwiad`.
        """
        # reset tensor tracker
        if self._is_valid_tensor(itm=self):
            self._tracker: Dict[Union[str, List[any], Tensor]] = {'opr': [], 'drv': [], 'chn': [], 'rlt': [], 'org': []}
        return None

    def instance_reset(self) -> None:
        r"""
        **Reset a Tensor.**

        ----------------------------------------------------------------------------------------------------------------

        ...

        ----------------------------------------------------------------------------------------------------------------

        Raises:
            UserWarning: If the function is used on a deleted Tensor.
                Turned off by toggling ikwiad.
                See :func:`Tensor.ikwiad`.
        """
        # reset tensor
        if self._is_valid_tensor(itm=self):
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
    def reset(cls, *args: Union['Tensor', str, int]) -> None:
        # find saved instances
        args = list(args)
        for i, arg in enumerate(args):
            _, args[i] = cls._get_tensor_reference(arg)
        # reset non-saved instance
        non_saved = [itm for i, itm in enumerate(cls._instances) if i not in args and itm is not None]
        for instance in non_saved:
            instance.instance_reset()
        return None

    @classmethod
    def grad_reset(cls) -> None:
        # turn on ikwiad
        user_ikwiad = Tensor._ikwiad
        Tensor._ikwiad = True
        for instance in cls._instances:
            if instance is None:
                continue
            if instance.type == 'grad':
                # reset gradients
                instance.instance_reset()
            # reset tensor tracker
            instance.instance_grad_reset()
        # return to user ikwiad
        Tensor._ikwiad = user_ikwiad
        return None

    @classmethod
    def instance_replace(cls, replaced: Union['Tensor', str, int], replacer: Union['Tensor', str, int]) -> None:
        # get tensors and ids
        replaced, replaced_id = cls._get_tensor_reference(itm=replaced)
        replacer, replacer_id = cls._get_tensor_reference(itm=replacer)

        # reset replaced tensor
        replaced.instance_reset()
        # move replacer tensor
        replacer._id = replaced_id
        cls._instances[replaced_id] = replacer
        cls._instances[replacer_id] = None
        return None

    @classmethod
    def zero_grad(cls, *args: Union['Tensor', str, int]) -> None:
        # reset instances
        cls.reset(*args)
        # reset gradient tracking
        cls.grad_reset()

    @classmethod
    def ikwiad(cls, ikwiad: bool) -> None:
        cls._ikwiad = bool(ikwiad)

    r"""
    Convenience methods.
    """

    @staticmethod
    def _is_valid_tensor(itm: 'Tensor') -> bool:
        if itm._type == 'deleted':
            # deleted Tensor
            if not Tensor._ikwiad:
                warn("Detected deleted Tensor reference", UserWarning)
            return False
        else:
            # non-deleted Tensor
            return True

    @classmethod
    def _get_tensor_reference(cls, itm: Union['Tensor', str, int]) -> Tuple['Tensor', int]:
        # turn on ikwiad
        user_ikwiad = Tensor._ikwiad
        Tensor._ikwiad = True
        # attempt hex conversion
        try:
            itm = int(itm, 16)
        except TypeError:
            pass
        except ValueError:
            pass

        if isinstance(itm, Tensor):
            # use object reference
            itm_id = itm._id
        elif isinstance(itm, int):
            # check index reference
            if len(cls._instances) <= itm:
                raise ValueError(
                    "Reference brought up outside Tensor instance list"
                    f"Currently, instance list only contains {len(cls._instances)} items"
                    f"A reference has been made to the {itm} index"
                )
            # use index reference
            itm_id = itm
            itm = cls._instances[itm_id]
        else:
            # invalid reference
            raise TypeError("Invalid Tensor reference type")
        if not cls._is_valid_tensor(itm=itm):
            # invalid tensor
            raise TypeError("Reference brought up with deleted Tensors")

        # return to user ikwiad
        Tensor._ikwiad = user_ikwiad
        # return items
        return itm, itm_id

    @classmethod
    def _add_instance(cls, itm: 'Tensor') -> None:
        try:
            cls._instances[cls._instances.index(None)] = itm
            itm._id = cls._instances.index(None)
        except ValueError:
            cls._instances.append(itm)
            itm._id = len(cls._instances) - 1
        return None

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
            raise NotImplementedError("'chain' has not been implemented in a subclass")

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
            raise NotImplementedError("'forward' has not been implemented in a subclass")

        @staticmethod
        def backward(main: np.ndarray) -> np.ndarray:
            r"""
            Backward function.

            Args:
                main (np.ndarray): Main array.

            Returns:
                np.ndarray: Result.
            """
            raise NotImplementedError("'backward' has not been implemented in a subclass")

        def call(self, main: 'Tensor') -> 'Tensor':
            r"""
            Function call.

            Args:
                main (Tensor): Non-deleted input Tensor.

            Returns:
                Tensor: Output Tensor.
            """
            # check tensor
            if not (isinstance(main, Tensor)) or main._type == 'deleted':
                raise TypeError("'main' must be a non-deleted Tensor")
            # calculate result
            result = Tensor(self.forward(main._tensor))
            result._tracker['org'] = [main, None]
            # track main
            self._update_track(
                obj=main,
                opr=f"{self._prefix}",
                drv=self.backward,
                chn=self.chain,
                rlt=[None, result]
            )
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
            raise NotImplementedError("'forward' has not been implemented in a subclass")

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
            raise NotImplementedError("'backward' has not been implemented in a subclass")

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
            raise NotImplementedError("'backward_o' has not been implemented in a subclass")

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
            raise NotImplementedError("'chain_o' has not been implemented in a subclass")

        def call(self, main: 'Tensor', other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
            r"""
            Function call.

            Args:
                main (Tensor): Non-deleted input Tensor.
                other (Tensor): Second non-deleted input Tensor.

            Returns:
                Tensor: Output Tensor.
            """
            # check tensor
            if not (isinstance(main, Tensor)) or main._type == 'deleted':
                print(type(main))
                print(main._tensor)
                print(main._type)  # this is the issue, I don't really know why :(
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
            self._update_track(
                obj=main,
                opr=f"{self._prefix}",
                drv=self.backward,
                chn=self.chain,
                rlt=[other, result]
            )
            if isinstance(other, Tensor):
                # track other
                self._update_track(
                    obj=other,
                    opr=f"{self._prefix}_o",
                    drv=self.backward_o,
                    chn=self.chain_o,
                    rlt=[main, result]
                )
            # return result
            return result

    @staticmethod
    def initializer_method(func: Callable) -> Callable:
        def wrapper(*args: int) -> 'Tensor':
            # check dimensions
            if len(args) != 2:
                raise ValueError("Initialization must occur with 2 dimensions")
            if not all(isinstance(arg, int) and 0 < arg for arg in args):
                raise ValueError("Each dimension must be a positive integer")
            # initialize tensor
            return Tensor(func(*args))
        return wrapper

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
            # NB: This method only relates the first relation instance.
            # If two alternate paths merge onto a single path, the binary search tree will stop before reaching the
            # second instance.
            # Implementing merging path functionality will slightly reduce relation speed and require some rewrites.
            # At this current time, adding the gradients seems to be the method.
            # If implemented, this will need to be checked for edge cases that may or may not exist where the
            # gradients are of different sizes.
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
                    [_relate(item=origin, target=target, trace=trace) for origin in origins]

        # relate tensors
        _relate(wrt, grad)
        if not isinstance(relation, list):
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
                res = res * np.eye(res.shape[0])
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

        # calculate initial gradient
        result = _derive(down=relation[-2], up=relation[-1])
        del relation[-1]
        while 1 < len(relation):
            # chain rule gradients
            result = Tensor.chain(down=_derive(down=relation[-2], up=relation[-1]), up=result)
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
            result = Tensor(obj=up._tracker['chn'][0][0](down._tensor, up._tensor), _gradient_override=True)
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
            ...

        @staticmethod
        def backward_o(other: np.ndarray, main: np.ndarray) -> np.ndarray:
            ...

        @staticmethod
        def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down @ up

        @staticmethod
        def chain_o(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down @ up

        def __call__(self, main: 'Tensor', other: 'Tensor') -> 'Tensor':
            return self.call(main, other)

    matmul = _MatMul()

    def __matmul__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        r"""Matrix multiplication of two Tensors."""
        return self.matmul(self, other)

    class _Pow(PairedTensorMethod):
        r"""Hadamard power built-in method."""
        def __init__(self):
            super().__init__(prefix="pow")

        @staticmethod
        def forward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return main ** other

        @staticmethod
        def backward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return other * (main ** (other - 1))

        @staticmethod
        def backward_o(other: np.ndarray, main: np.ndarray) -> np.ndarray:
            return np.log(main) * (main ** other)

        @staticmethod
        def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down @ up

        @staticmethod
        def chain_o(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down @ up

        def __call__(self, main: 'Tensor', other: 'Tensor') -> 'Tensor':
            return self.call(main, other)

    pow = _Pow()

    def __pow__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        r"""Hadamard power of two Tensors."""
        return pow(self, other)

    class _Mul(PairedTensorMethod):
        r"""Hadamard multiplication built-in method."""
        def __init__(self):
            super().__init__(prefix="mul")

        @staticmethod
        def forward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return main * other

        @staticmethod
        def backward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return other * (main * 0.0 + 1.0)

        @staticmethod
        def backward_o(other: np.ndarray, main: np.ndarray) -> np.ndarray:
            return main * (other * 0.0 + 1.0)

        @staticmethod
        def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down @ up

        @staticmethod
        def chain_o(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down @ up

        def __call__(self, main: 'Tensor', other: 'Tensor') -> 'Tensor':
            return self.call(main, other)

    mul = _Mul()

    def __mul__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        r"""Hadamard multiplication of two Tensors"""
        return self.mul(self, other)

    class _TrueDiv(PairedTensorMethod):
        r"""Hadamard division built-in method."""
        def __init__(self):
            super().__init__(prefix="truediv")

        @staticmethod
        def forward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return main / other

        @staticmethod
        def backward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return other * (main * 0.0 + 1.0)

        @staticmethod
        def backward_o(other: np.ndarray, main: np.ndarray) -> np.ndarray:
            return (other * 0.0 + 1.0) / main

        @staticmethod
        def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down @ up

        @staticmethod
        def chain_o(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down @ up

        def __call__(self, main: 'Tensor', other: 'Tensor') -> 'Tensor':
            return self.call(main, other)

    truediv = _TrueDiv()

    def __truediv__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        r"""Hadamard division of two Tensors."""
        return self.truediv(self, other)

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
            return down @ up

        @staticmethod
        def chain_o(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down @ up

        def __call__(self, main: 'Tensor', other: 'Tensor') -> 'Tensor':
            return self.call(main, other)

    add = _Add()

    def __add__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        r"""Addition of two Tensors."""
        return self.add(self, other)

    class _Sub(PairedTensorMethod):
        r"""Mathematical subtraction method."""
        def __init__(self):
            super().__init__(prefix="sub")

        @staticmethod
        def forward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return main - other

        @staticmethod
        def backward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return main * 0.0 + 1.0

        @staticmethod
        def backward_o(other: np.ndarray, main: np.ndarray) -> np.ndarray:
            return other * 0.0 - 1.0

        @staticmethod
        def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down @ up

        @staticmethod
        def chain_o(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            return down @ up

        def __call__(self, main: 'Tensor', other: 'Tensor') -> 'Tensor':
            return self.call(main, other)

    sub = _Sub()

    def __sub__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        r"""Subtraction of two Tensors."""
        return self.sub(self, other)
