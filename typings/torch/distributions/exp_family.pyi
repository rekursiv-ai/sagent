from torch.distributions.distribution import Distribution

__all__ = ["ExponentialFamily"]

class ExponentialFamily(Distribution):
    def entropy(self) -> Tensor | float: ...
