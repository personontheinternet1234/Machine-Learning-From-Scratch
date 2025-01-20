r"""Built-in models."""

from functional.objects import Tensor


class Base:
    def __init__(self, *, status: bool = False, ikwiad: bool = True):
        self._status = bool(status)
        self._ikwiad = bool(ikwiad)

    def _forward(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError("Forward pass hasn't been configured in a subclass")

    def _backward(self, y: Tensor, yhat: Tensor) -> Tensor:
        raise NotImplementedError("Backward pass hasn't been configured in a subclass")

    def _optimize(self, theta: Tensor, nabla: Tensor) -> Tensor:
        raise NotImplementedError("Optimization hasn't been configured in a subclass")

    def _step(self, x: Tensor, y: Tensor) -> Tensor:
        yhat = self._forward(x=x, y=y)
        nablas = self._backward(y=y, yhat=yhat)
