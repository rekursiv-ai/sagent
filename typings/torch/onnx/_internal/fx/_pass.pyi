import abc
import dataclasses

from torch._subclasses import fake_tensor

import torch
import torch.fx

@dataclasses.dataclass
class PackageInfo:
    package_name: str
    version: str | None
    commit_hash: str | None
    def to_onnx_domain_string(self) -> str: ...
    @classmethod
    def from_python_class(cls, python_class_name: type | str) -> PackageInfo: ...

@dataclasses.dataclass
class GraphModuleOnnxMeta:
    package_info: PackageInfo

def maybe_fx_graph_tabular(graph: torch.fx.Graph) -> str | None: ...

class Transform(abc.ABC):
    module: torch.fx.GraphModule
    fake_mode: fake_tensor.FakeTensorMode | None
    def __init__(self, module: torch.fx.GraphModule) -> None: ...
    def run(self, *args, **kwargs) -> torch.fx.GraphModule: ...
