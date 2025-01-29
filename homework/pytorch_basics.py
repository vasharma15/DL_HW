import torch

class PyTorchBasics:
    @staticmethod
    def make_it_pytorch_1(x: torch.Tensor) -> torch.Tensor:
        return x[::3]

    @staticmethod
    def make_it_pytorch_2(x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=2).values

    @staticmethod
    def make_it_pytorch_3(x: torch.Tensor) -> torch.Tensor:
        return torch.unique(x)

    @staticmethod
    def make_it_pytorch_4(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (y > x.mean()).sum()

    @staticmethod
    def make_it_pytorch_5(x: torch.Tensor) -> torch.Tensor:
        return x.mT

    @staticmethod
    def make_it_pytorch_6(x: torch.Tensor) -> torch.Tensor:
        return x.diagonal()

    @staticmethod
    def make_it_pytorch_7(x: torch.Tensor) -> torch.Tensor:
        return x.flip(1).diagonal()

    @staticmethod
    def make_it_pytorch_8(x: torch.Tensor) -> torch.Tensor:
        return x.cumsum(dim=0)

    @staticmethod
    def make_it_pytorch_9(x: torch.Tensor) -> torch.Tensor:
        return x.cumsum(dim=0).cumsum(dim=1)

    @staticmethod
    def make_it_pytorch_10(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.where(x < c, torch.zeros_like(x), x)

    @staticmethod
    def make_it_pytorch_11(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.nonzero(x < c).T

    @staticmethod
    def make_it_pytorch_12(x: torch.Tensor, m: torch.BoolTensor) -> torch.Tensor:
        return x[m]
