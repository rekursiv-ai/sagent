from collections.abc import Callable, Sequence
from typing import Any, Literal

from torch._inductor.utils import InputType
from torch.compiler._cache import CacheInfo
from torch.fx import GraphModule

log = ...

class CompiledArtifact:
    _compiled_fn: Callable[..., Any]
    _artifacts: tuple[bytes, CacheInfo] | None
    def __init__(
        self, compiled_fn: Callable[..., Any], artifacts: tuple[bytes, CacheInfo] | None
    ) -> None: ...
    def __call__(self, *args: Any) -> Any: ...
    def save(
        self, *, path: str, format: Literal["binary", "unpacked"] = ...
    ) -> None: ...
    @staticmethod
    def load(
        *, path: str, format: Literal["binary", "unpacked"] = ...
    ) -> CompiledArtifact: ...

def standalone_compile(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    *,
    dynamic_shapes: Any,
    options: Any,
) -> CompiledArtifact: ...
