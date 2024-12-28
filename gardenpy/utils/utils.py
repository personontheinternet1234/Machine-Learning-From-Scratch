r"""
utils.py

Includes utility functions for GardenPy.
Contains a parameter checker.
"""

from typing import Optional
from types import LambdaType
import warnings

from .errors import (
    MissingMethodError
)


class ParamChecker:
    r"""
    ParamChecker is a parameter checker for any parameters.

    By specifying the default values, their datatypes, their value types, and their conversion types, you can set a
    reusable parameter checker that replaces all unfilled parameters with a default value, checks their datatype, checks
    if they match a conditional, and converts them to a certain value.
    """
    def __init__(self, name: str = 'Parameters', *, ikwiad: bool = False):
        r"""
        Initializes the ParamChecker class.

        Args:
            name (str, optional):
                Name of the ParamChecker instance.
                Used when an error occurs for ease error traceback.
                Defaults to "Parameters".
            ikwiad (bool):
                "I know what I am doing" (ikwiad).
                If True, removes all warning messages.
                Defaults to False.
        """
        # warning settings
        self._name = str(name)
        self._ikwiad = bool(ikwiad)

        # instantiation checker
        self._is_set = False

        # internal checkers
        self._default = None
        self._dtypes = None
        self._vtypes = None
        self._ctypes = None

    def _validate_dict(self, param_dict, name, check_callable=False, check_lambda=False):
        # validate dictionary
        if not isinstance(param_dict, dict):
            raise TypeError(f"'{name}' in {self._name} must be a dict")
        for key, value in param_dict.items():
            if check_lambda and not isinstance(value, LambdaType):
                raise TypeError(f"Invalid lambda in '{name}' in {self._name}: {key} -> {value}")
            if check_callable and callable(value):
                raise TypeError(f"Callable not allowed in '{name}' in {self._name}: {key} -> {value}")

    def set_types(
            self,
            default: Optional[dict] = None,
            dtypes: Optional[dict] = None,
            vtypes: Optional[dict] = None,
            ctypes: Optional[dict] = None
    ) -> None:
        r"""
        Sets the internal types of the parameter checker.

        Args:
            default (dict):
                Default parameter values.
            dtypes (dict):
                Possible datatypes of parameters.
            vtypes (dict):
                Lambda functions for conditionals of inputted parameters.
            ctypes (dict):
                Lambda functions for conversion types of passed parameters.

        Returns:
            None

        Raises:
            TypeError: If dictionaries were of the wrong type.
            ValueError: If dictionary keys don't match.
        """
        if default is None:
            # default none
            return None

        # validate dicts
        self._validate_dict(default, 'default', check_callable=True)
        self._validate_dict(dtypes, 'dtypes')
        self._validate_dict(vtypes, 'vtypes', check_lambda=True)
        self._validate_dict(ctypes, 'ctypes', check_lambda=True)

        # check for key matching
        keys = [default.keys(), dtypes.keys(), vtypes.keys(), ctypes.keys()]
        if not all(k == keys[0] for k in keys):
            raise ValueError(f"Keys don't match for '{self._name}'")

        # set internal checkers
        self._default = default
        self._dtypes = dtypes
        self._vtypes = vtypes
        self._ctypes = ctypes
        self._is_set = True
        return None

    def check_params(self, params: Optional[dict] = None, **kwargs) -> dict:
        r"""
        Checks parameters.

        Args:
            params (dict, optional):
                Parameters to be checked.
            **kwargs:
                kwargs of the parameters to be checked.

        Returns:
            dict:
                The checked parameters.

        Raises:
            MissingMethodError: If default values weren't set.
            TypeError: If any parameters were of the wrong type.
            ValueError: If invalid values were passed for any of the parameters.
        """
        # check for default parameters
        if not self._is_set:
            raise MissingMethodError(f"Default parameters not set for '{self._name}'")

        # initialize as default
        final = self._default.copy()

        if params is None and kwargs is None:
            # return default
            return final

        # set params
        if params and not isinstance(params, dict):
            raise TypeError(f"'params' in {self._name} must be a dictionary")
        params = params if params else {}
        if kwargs:
            params.update(kwargs)

        for key, prm in params.items():
            if key not in self._default and self._ikwiad:
                # invalid key and warning
                print()
                warnings.warn(
                    f"\nInvalid parameter for '{self._name}': '{key}'\n"
                    f"Choose from: '{[pos for pos in self._default]}'",
                    UserWarning
                )
                continue
            elif key not in self._default:
                # invalid key
                continue

            # datatype check
            if not isinstance(prm, self._dtypes[key]):
                raise ValueError(
                    f"Invalid datatype for '{self._name}' '{key}': '{prm}'\n"
                    f"Choose from: {self._dtypes[key]}"
                )
            if not self._vtypes[key](prm):
                raise ValueError(
                    f"Invalid value for '{self._name}' '{key}': '{prm}'\n"
                    f"Failed conditional: {self._vtypes[key]}"
                )
            # set parameter
            final[key] = self._ctypes[key](prm)

        # return parameters
        return final
