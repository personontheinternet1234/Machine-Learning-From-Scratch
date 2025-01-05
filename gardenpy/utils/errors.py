r"""Built-in errors."""

from typing import Optional


class MissingMethodError(Exception):
    r"""
    **When a method hasn't been set.**
    """
    pass


class TrackingError(Exception):
    r"""
    **When Tensor tracking is unsuccessful.**
    """
    def __init__(self, grad, wrt, *, message: Optional[str] = None):
        r"""
        **Tracking error instance.**

        ----------------------------------------------------------------------------------------------------------------

        Args:
            grad (Tensor): First Tensor relating to the tracking error.
            wrt (Tensor): Second Tensor relating to the tracking error.
            message (str, optional): The error message.

        Note:
            A built-in error message reports common information about the tracking error if no message is given.
        """
        # message check
        # error message
        if message is None:
            message = (
                f"No relation could be found between \n {grad} \n and \n {wrt} \n"
                "This might be due to no clear relation between the Tensors, "
                "accidental clearing of trackers, "
                "deletion of Tensors, "
                "or accidental reference to the wrong Tensor"
            )
        super().__init__(str(message))
