from typing import Any

from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.functional_tensor import BaseFunctionalizeAPI
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

import torch

class RunConstGraph(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(
        self, graph: torch.fx.GraphModule, args: tuple[object, ...]
    ) -> object: ...

run_const_graph = ...

@run_const_graph.py_impl(ProxyTorchDispatchMode)
def run_const_graph_dispatch_mode(
    mode: ProxyTorchDispatchMode, graph: torch.fx.GraphModule, args: tuple[object, ...]
) -> object: ...
@run_const_graph.py_functionalize_impl
def run_const_graph_functional(
    ctx: BaseFunctionalizeAPI, graph: torch.fx.GraphModule, args: tuple[Any, ...]
) -> Any: ...
@run_const_graph.py_impl(FakeTensorMode)
def run_const_graph_fake_tensor_mode(
    mode: FakeTensorMode, graph: torch.fx.GraphModule, args: tuple[object, ...]
) -> object: ...
@run_const_graph.py_impl(DispatchKey.CPU)
def run_const_graph_cpu(
    graph: torch.fx.GraphModule, args: tuple[object, ...]
) -> object: ...
