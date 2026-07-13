import abc

import torch

__all__ = ["WeightedQuantizedModule"]

class WeightedQuantizedModule(torch.nn.Module, metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def from_reference(cls, ref_module, output_scale, output_zero_point): ...

_pair_from_first = ...
