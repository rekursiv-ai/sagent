import abc

from torch import nn

class Sharder(abc.ABC):
    @abc.abstractmethod
    def shard(self, module: nn.Module) -> nn.Module: ...
