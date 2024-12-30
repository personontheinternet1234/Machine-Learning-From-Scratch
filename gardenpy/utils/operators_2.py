from .objects_2 import Tensor


def tensor(obj: any) -> Tensor:
    return Tensor(obj)


def nabla(grad: Tensor, wrt: Tensor) -> Tensor:
    return Tensor.nabla(grad, wrt)


def chain(down: Tensor, up: Tensor) -> Tensor:
    return Tensor.chain(down, up)
