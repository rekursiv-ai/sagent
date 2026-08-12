from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

import torch.fx.graph

class CustomGraphPass(ABC):
    @abstractmethod
    def __call__(self, graph: torch.fx.graph.Graph) -> None: ...
    @abstractmethod
    def uuid(self) -> None: ...

class CustomGraphModulePass(ABC):
    @abstractmethod
    def __call__(self, gm: torch.fx.GraphModule) -> None: ...
    @abstractmethod
    def uuid(self) -> None: ...

type CustomGraphPassType = (
    CustomGraphPass | Callable[[torch.fx.graph.Graph], None] | None
)

@lru_cache(1)
def get_hash_for_files(paths: tuple[str], extra: str = ...) -> bytes: ...

class CustomPartitionerFn(ABC):
    @abstractmethod
    def __call__(
        self, gm: torch.fx.GraphModule, joint_inputs: Sequence[object], **kwargs: Any
    ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]: ...
    @abstractmethod
    def uuid(self) -> None: ...

type CustomPartitionerFnType = CustomPartitionerFn | None
