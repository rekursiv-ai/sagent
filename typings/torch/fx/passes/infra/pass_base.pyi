from collections import namedtuple

import abc

from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule

__all__ = ["PassBase", "PassResult"]

@compatibility(is_backward_compatible=False)
class PassResult(namedtuple("PassResult", ["graph_module", "modified"])):
    __slots__ = ...
    def __new__(cls, graph_module, modified) -> Self: ...

@compatibility(is_backward_compatible=False)
class PassBase(abc.ABC):
    def __call__(self, graph_module: GraphModule) -> PassResult | None: ...
    @abc.abstractmethod
    def call(self, graph_module: GraphModule) -> PassResult | None: ...
    def requires(self, graph_module: GraphModule) -> None: ...
    def ensures(self, graph_module: GraphModule) -> None: ...
