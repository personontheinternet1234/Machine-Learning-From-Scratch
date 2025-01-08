r"""Utility functions."""

from typing import Callable, Dict, Union
from types import LambdaType
import warnings


class Params:
    def __init__(
            self,
            default: Union[Dict[str, Union[int, float, str, bool]], None],
            dtypes: Union[Dict[str, Union[tuple, any]], None],
            vtypes: Union[Dict[str, Callable], None],
            ctypes: Union[Dict[str, Callable], None]
    ):
        self._default = self._check_default(default)
        self._dtypes = self._check_dtypes(dtypes)
        self._vtypes = self._check_vtypes(vtypes)
        self._ctypes = self._check_ctypes(ctypes)

    def __repr__(self):
        return (
            f"default: {self._default}\n"
            f"dtypes: {self._dtypes}\n"
            f"vtypes: {self._vtypes}\n"
            f"ctypes: {self._ctypes}"
        )

    @staticmethod
    def _check_type(itm):
        if itm is not None and not isinstance(itm, dict):
            # check itm
            raise TypeError("Each parameter item must be None or a dictionary")
        # return itm
        return itm

    @staticmethod
    def _check_default(default: Union[dict, None]) -> Union[dict, None]:
        # check dict
        Params._check_type(default)
        # check default
        if (
                isinstance(default, dict) and
                not all([isinstance(itm, (int, float, str, bool)) for itm in default.values()])
        ):
            raise TypeError("Some items in 'default' weren't of the accepted type")
        # return default
        return default

    @staticmethod
    def _check_dtypes(dtypes: Union[dict, None]) -> Union[dict, None]:
        # check dict
        Params._check_type(dtypes)
        # check dtypes
        if (
                isinstance(dtypes, dict) and
                not all([itm is not callable for itm in dtypes.values()])
        ):
            raise TypeError("Some items in 'dtypes' weren't of the accepted type")
        # return dtypes
        return dtypes

    @staticmethod
    def _check_vtypes(vtypes: Union[dict, None]) -> Union[dict, None]:
        # check dict
        Params._check_type(vtypes)
        # check vtypes
        if (
                isinstance(vtypes, dict) and
                not all([isinstance(itm, LambdaType) for itm in vtypes.values()])
        ):
            raise TypeError("Some items in 'vtypes' weren't of the accepted type")
        # return vtypes
        return vtypes

    @staticmethod
    def _check_ctypes(ctypes: Union[dict, None]) -> Union[dict, None]:
        # check dict
        Params._check_type(ctypes)
        # check ctypes
        if (
                isinstance(ctypes, dict) and
                not all([isinstance(itm, LambdaType) for itm in ctypes.values()])
        ):
            raise TypeError("Some items in 'ctypes' weren't of the accepted type")
        # return ctypes
        return ctypes

    @property
    def default(self) -> Union[Dict[str, Union[int, float, str, bool]], None]:
        return self._default

    @property
    def dtypes(self) -> Union[Dict[str, Union[tuple, any]], None]:
        return self._dtypes

    @property
    def vtypes(self) -> Union[Dict[str, Callable], None]:
        return self._vtypes

    @property
    def ctypes(self) -> Union[Dict[str, Callable], None]:
        return self._ctypes


class ParamChecker:
    _none_params = Params(default=None, dtypes=None, vtypes=None, ctypes=None)

    def __init__(self, prefix: str = 'Parameters', parameters: Params = _none_params, *, ikwiad: bool = False):
        self._prefix = str(prefix)
        self._ikwiad = bool(ikwiad)
        self._params = self._validate_params(params=parameters)

    @property
    def parameters(self):
        return self._params

    def _validate_dict(
            self,
            param_dict: dict,
            name: str,
            check_callable:
            bool = False,
            check_lambda: bool = False
    ) -> None:
        # validate dictionary
        if not isinstance(param_dict, dict):
            raise TypeError(f"'{name}' in {self._prefix} must be a dict")
        for key, value in param_dict.items():
            if check_lambda and not isinstance(value, LambdaType):
                raise TypeError(f"Invalid lambda in '{name}' in {self._prefix}: {key} -> {value}")
            if check_callable and callable(value):
                raise TypeError(f"Callable not allowed in '{name}' in {self._prefix}: {key} -> {value}")
        return None

    def _validate_params(self, params: Params) -> Params:
        if not isinstance(params, Params):
            raise TypeError("'params' must be Params")

        if params.default is None:
            # default none
            self._is_set = True
            return params

        # validate dicts
        self._validate_dict(params.default, 'default', check_callable=True)
        self._validate_dict(params.dtypes, 'dtypes')
        self._validate_dict(params.vtypes, 'vtypes', check_lambda=True)
        self._validate_dict(params.ctypes, 'ctypes', check_lambda=True)

        # check for key matching
        keys = [params.default.keys(), params.dtypes.keys(), params.vtypes.keys(), params.ctypes.keys()]
        if not all(k == keys[0] for k in keys):
            raise ValueError(f"Keys don't match for '{self._prefix}'")

        return params

    def check_params(self, params: Union[dict, None] = None, **kwargs) -> Union[dict, None]:
        r"""
        **Checks parameters.**

        ----------------------------------------------------------------------------------------------------------------

        Args:
            params (dict, optional): Parameters to be checked.
            **kwargs: Key-word args of the parameters to be checked.

        Returns:
            dict | NoneType: The checked parameters.
                Returns None of no parameters are taken.

        Raises:
            TypeError: If any parameters were of the wrong type.
            ValueError: If invalid values were passed for any of the parameters.
        """
        # check for no parameters
        if self._params.default is None:
            return None

        # initialize as default
        final = self._params.default.copy()

        if params is None and kwargs is None:
            # return default
            return final

        # set params
        if params and not isinstance(params, dict):
            raise TypeError(f"'params' in {self._prefix} must be a dictionary")
        params = params if params else {}
        if kwargs:
            params.update(kwargs)

        for key, prm in params.items():
            if key not in self._params.default and self._ikwiad:
                # invalid key and warning
                print()
                warnings.warn(
                    f"\nInvalid parameter for '{self._prefix}': '{key}'\n"
                    f"Choose from: '{[pos for pos in self._params.default]}'",
                    UserWarning
                )
                continue
            elif key not in self._params.default:
                # invalid key
                continue

            # datatype check
            if not isinstance(prm, self._params.dtypes[key]):
                raise ValueError(
                    f"Invalid datatype for '{self._prefix}' '{key}': '{prm}'\n"
                    f"Choose from: {self._params.dtypes[key]}"
                )
            if not self._params.vtypes[key](prm):
                raise ValueError(
                    f"Invalid value for '{self._prefix}' '{key}': '{prm}'\n"
                    f"Failed conditional: {self._params.vtypes[key]}"
                )
            # set parameter
            final[key] = self._params.ctypes[key](prm)

        # return parameters
        return final


# class ParamCheckerOld:
#     r"""
#     ParamChecker is a parameter checker for any parameters.
#
#     By specifying the default values, their datatypes, their value types, and their conversion types, you can set a
#     reusable parameter checker that replaces all unfilled parameters with a default value, checks their datatype, checks
#     if they match a conditional, and converts them to a certain value.
#     """
#     def __init__(self, name: str = 'Parameters', *, ikwiad: bool = False):
#         r"""
#         Initializes the ParamChecker class.
#
#         Args:
#             name (str, optional), default = 'Parameters': Name of ParamChecker instance for error traceback.
#             ikwiad (bool), default = False: Remove all warning messages ("I know what I am doing" - ikwiad).
#         """
#         # warning settings
#         self._name = str(name)
#         self._ikwiad = bool(ikwiad)
#
#         # instantiation checker
#         self._is_set = False
#
#         # internal checkers
#         self._default = None
#         self._dtypes = None
#         self._vtypes = None
#         self._ctypes = None
#
#     def _validate_dict(self, param_dict, name, check_callable=False, check_lambda=False):
#         # validate dictionary
#         if not isinstance(param_dict, dict):
#             raise TypeError(f"'{name}' in {self._name} must be a dict")
#         for key, value in param_dict.items():
#             if check_lambda and not isinstance(value, LambdaType):
#                 raise TypeError(f"Invalid lambda in '{name}' in {self._name}: {key} -> {value}")
#             if check_callable and callable(value):
#                 raise TypeError(f"Callable not allowed in '{name}' in {self._name}: {key} -> {value}")
#
#     def set_types(
#             self,
#             default: Optional[dict] = None,
#             dtypes: Optional[dict] = None,
#             vtypes: Optional[dict] = None,
#             ctypes: Optional[dict] = None
#     ) -> None:
#         r"""
#         Sets the internal types of the parameter checker.
#
#         Args:
#             default (dict): Default parameter values.
#             dtypes (dict): Possible datatypes of parameters.
#             vtypes (dict): Lambda functions for conditionals of inputted parameters.
#             ctypes (dict): Lambda functions for conversion types of passed parameters.
#
#         Raises:
#             TypeError: If dictionaries were of the wrong type.
#             ValueError: If dictionary keys don't match.
#         """
#         if default is None:
#             # default none
#             self._is_set = True
#             return None
#
#         # validate dicts
#         self._validate_dict(default, 'default', check_callable=True)
#         self._validate_dict(dtypes, 'dtypes')
#         self._validate_dict(vtypes, 'vtypes', check_lambda=True)
#         self._validate_dict(ctypes, 'ctypes', check_lambda=True)
#
#         # check for key matching
#         keys = [default.keys(), dtypes.keys(), vtypes.keys(), ctypes.keys()]
#         if not all(k == keys[0] for k in keys):
#             raise ValueError(f"Keys don't match for '{self._name}'")
#
#         # set internal checkers
#         self._default = default
#         self._dtypes = dtypes
#         self._vtypes = vtypes
#         self._ctypes = ctypes
#         self._is_set = True
#         return None
#
#     def check_params(self, params: Optional[dict] = None, **kwargs) -> Union[dict, None]:
#         r"""
#         Check parameters.
#
#         Args:
#             params (dict, optional): Parameters to be checked.
#             **kwargs: kwargs of the parameters to be checked.
#
#         Returns:
#             dict: The checked parameters.
#
#         Raises:
#             MissingMethodError: If default values weren't set.
#             TypeError: If any parameters were of the wrong type.
#             ValueError: If invalid values were passed for any of the parameters.
#         """
#         # check for default parameters
#         if not self._is_set:
#             raise MissingMethodError(f"Default parameters not set for '{self._name}'")
#
#         # check for no parameters
#         if self._default is None:
#             return None
#
#         # initialize as default
#         final = self._default.copy()
#
#         if params is None and kwargs is None:
#             # return default
#             return final
#
#         # set params
#         if params and not isinstance(params, dict):
#             raise TypeError(f"'params' in {self._name} must be a dictionary")
#         params = params if params else {}
#         if kwargs:
#             params.update(kwargs)
#
#         for key, prm in params.items():
#             if key not in self._default and self._ikwiad:
#                 # invalid key and warning
#                 print()
#                 warnings.warn(
#                     f"\nInvalid parameter for '{self._name}': '{key}'\n"
#                     f"Choose from: '{[pos for pos in self._default]}'",
#                     UserWarning
#                 )
#                 continue
#             elif key not in self._default:
#                 # invalid key
#                 continue
#
#             # datatype check
#             if not isinstance(prm, self._dtypes[key]):
#                 raise ValueError(
#                     f"Invalid datatype for '{self._name}' '{key}': '{prm}'\n"
#                     f"Choose from: {self._dtypes[key]}"
#                 )
#             if not self._vtypes[key](prm):
#                 raise ValueError(
#                     f"Invalid value for '{self._name}' '{key}': '{prm}'\n"
#                     f"Failed conditional: {self._vtypes[key]}"
#                 )
#             # set parameter
#             final[key] = self._ctypes[key](prm)
#
#         # return parameters
#         return final
