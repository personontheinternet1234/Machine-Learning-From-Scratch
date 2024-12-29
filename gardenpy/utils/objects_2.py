r"""
objects.py

Includes GardenPy's objects.
Contains Tensor class.
"""

import numpy as np


class Tensor2:
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
        if not np.issubdtype(obj.dtype, np.number):
            raise TypeError("'obj' must only contain numerical values")

        # set tensor internals
        self._id = len(Tensor2._instances)
        self._tensor = obj
        self._type = 'mat'
        self._tracker = {
            'opr': [],
            'drv': [],
            'chn': [],
            'rlt': [],
            'org': [None]
        }

        # update instances
        Tensor2._instances.append(self)

    def __repr__(self):
        r"""
        Human-readable format of the tensor.

        Returns:
            str:
                NumPy array representation of Tensor.
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
            str | NoneType:
                Current Tensor ID.
        """
        return hex(self._id)

    @property
    def type(self):
        r"""
        Tensor type, ranging from 'mat' for matrices, 'grad' for gradients, and 'deleted' for deleted instances.

        Returns:
            str:
                Tensor type.
        """
        return self._type

    @property
    def tracker(self):
        r"""
        Gets the internal Tensor tracker, referencing pointers to each object or the Tensor ID if applicable.
        The tracker keeps track of the item ID, forward operations, derivative operations, chain-rule operations, object
        relationships, and origin relationships.

        Returns:
            dict:
                Tensor tracker.
        """
        alt_tracker = self._tracker
        alt_tracker['rlt'] = [
            [itm[0].id if isinstance(itm[0], Tensor2) else hex(id(itm[0])),
             itm[1].id if isinstance(itm[1], Tensor2) else hex(id(itm[1]))]
            for itm in alt_tracker['rlt']
        ]
        alt_tracker['org'] = [
            itm.id if isinstance(itm, Tensor2) else hex(id(itm))
            for itm in alt_tracker['org']
        ]
        return {'itm': self.id, **alt_tracker}

    @property
    def array(self):
        r"""
        Tensor NumPy Value.

        Returns:
            np.ndarray:
                NumPy array value of Tensor.
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
            'org': [None]
        }
        return None

    def instance_reset(self):
        # reset tensor
        Tensor2._instances[self._id] = None
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
        if isinstance(replaced, Tensor2) and replaced.type != 'deleted':
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
        if isinstance(replacer, Tensor2) and replacer.type != 'deleted':
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
            # prefix set
            self._prefix = prefix

        @staticmethod
        def _update_track(obj, opr, drv, chn, rlt):
            # tracker update
            assert isinstance(obj, Tensor2)
            obj._tracker['opr'].append(opr)
            obj._tracker['drv'].append(drv)
            obj._tracker['chn'].append(chn)
            obj._tracker['rlt'].append(rlt)
            return None

        @staticmethod
        def forward(main, other):
            # forward method
            raise NotImplementedError("'forward' must be implemented in a subclass")

        @staticmethod
        def backward(main, other):
            # backward method
            raise NotImplementedError("'backward' must be implemented in a subclass")

        @staticmethod
        def chain(down, up):
            # chain method
            raise NotImplementedError("'chain' must be implemented in a subclass")

    class LoneTensorMethod(TensorMethod):
        def call(self, main):
            # check tensor
            if not (isinstance(main, Tensor2)) or main.type == 'deleted':
                raise TypeError("'main' must be a non-deleted Tensor")
            # calculate result
            result = Tensor2(self.forward(main._tensor, None))
            result._tracker['org'] = [main, None]
            # track main
            self._update_track(main, f"{self._prefix}", self.backward, self.chain, [None, result])
            # return result
            return result

    class PairedTensorMethod(TensorMethod):
        @staticmethod
        def backward_o(other, main):
            # other backward method
            raise NotImplementedError("'backward_o' must be implemented in a subclass")

        @staticmethod
        def chain_o(down, up):
            # other chain method
            raise NotImplementedError("'chain_o' must be implemented in a subclass")

        def call(self, main, other):
            # check tensor
            if not (isinstance(main, Tensor2)) or main.type == 'deleted':
                raise TypeError("'main' must be a non-deleted Tensor")
            # set array
            if isinstance(other, Tensor2):
                arr = other._tensor
            else:
                arr = other
            # calculate result
            result = Tensor2(self.forward(main._tensor, arr))
            result._tracker['org'] = [main, other]
            # track main
            self._update_track(main, f"{self._prefix}", self.backward, self.chain, [other, result])
            if isinstance(other, Tensor2):
                # track other
                self._update_track(other, f"{self._prefix}_o", self.backward_o, self.chain_o, [main, result])
            # return result
            return result

    ####################################################################################################################
    # Built-in methods.
    ####################################################################################################################

    class Pow(PairedTensorMethod):

        def __init__(self):
            # prefix set
            super().__init__(prefix="pow")

        @staticmethod
        def forward(main, other):
            r"""
            Forward operation of power.

            Mathematical Representation:
                $$\m^o$$

            Args:
                main (np.ndarray):
                    Main object instance.
                other (np.ndarray):
                    Other object instance.
            """
            return main ** other

        @staticmethod
        def backward(main, other):
            r"""
            Backward operation of power.

            Mathematical Representation:
                $$om^{o - 1}$$

            Args:
                main (np.ndarray):
                    Main object instance.
                other (np.ndarray):
                    Other object instance.
            """
            return other * (main ** (other - 1)) * np.eye(main.shape[1])

        @staticmethod
        def backward_o(other, main):
            r"""
            Other backward operation of power.

            Mathematical Representation:
                $$\ln{m} m^o$$

            Args:
                main (np.ndarray):
                    Main object instance.
                other (np.ndarray):
                    Other object instance.
            """
            return np.log(main) * (main ** other) * np.eye(other.shape[1])

        @staticmethod
        def chain(down, up):
            r"""
            Chain operation of power.

            Mathematical Representation:
                $$du$$

            Args:
                down (np.ndarray):
                    downstream gradient.
                up (np.ndarray):
                    upstream gradient.
            """
            return down * up

        @staticmethod
        def chain_o(down, up):
            r"""
            Other chain operation of power.

            Mathematical Representation:
                $$du$$

            Args:
                down (np.ndarray):
                    downstream gradient.
                up (np.ndarray):
                    upstream gradient.
            """
            return down * up

        def __call__(self, main, other):
            r"""
            ...
            """
            return self.call(main, other)

    ####################################################################################################################
    # Callable forward methods.
    ####################################################################################################################

    def __pow__(self, other):
        # pow dunder method
        return Tensor2.Pow()(self, other)

    ####################################################################################################################
    # Gradient calculation methods.
    ####################################################################################################################

    ...
