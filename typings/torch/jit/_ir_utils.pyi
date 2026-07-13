import types

import torch

class _InsertPoint:
    def __init__(
        self,
        insert_point_graph: torch._C.Graph,
        insert_point: torch._C.Node | torch._C.Block,
    ) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None: ...

def insert_point_guard(
    self: torch._C.Graph, insert_point: torch._C.Node | torch._C.Block
) -> _InsertPoint: ...
