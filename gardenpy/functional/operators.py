r"""External object operator references."""

from functools import wraps

from .objects import Tensor


@wraps(wrapped=Tensor.__init__)
def tensor(obj: any) -> Tensor:
    return Tensor(obj=obj)


@wraps(wrapped=Tensor.nabla)
def nabla(grad: Tensor, wrt: Tensor, *, binary: bool = True) -> Tensor:
    return Tensor.nabla(grad=grad, wrt=wrt, binary=binary)


@wraps(wrapped=Tensor.chain)
def chain(down: Tensor, up: Tensor) -> Tensor:
    return Tensor.chain(down=down, up=up)
