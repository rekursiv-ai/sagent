from collections.abc import Callable
from typing import Any

def chunk_layer(
    layer: Callable,
    inputs: dict[str, Any],
    chunk_size: int,
    no_batch_dims: int,
    low_mem: bool = ...,
    _out: Any = ...,
    _add_into_out: bool = ...,
) -> Any: ...

class ChunkSizeTuner:
    def __init__(self, max_chunk_size: int = ...) -> None: ...
    def tune_chunk_size(
        self, representative_fn: Callable, args: tuple, min_chunk_size: int
    ) -> int: ...
