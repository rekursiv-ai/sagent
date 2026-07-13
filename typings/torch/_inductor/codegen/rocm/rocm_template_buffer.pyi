from collections.abc import Callable, Sequence
from typing import TypeVar
from typing_extensions import ParamSpec

from ...ir import Buffer, Layout, TemplateBuffer

_P = ParamSpec("_P")
_T = TypeVar("_T")

class ROCmTemplateBuffer(TemplateBuffer):
    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[Buffer],
        make_kernel_render: Callable[_P, _T],
        workspace_size: int,
        template: ROCmTemplate,
    ) -> None: ...
    def get_workspace_size(self) -> int: ...
