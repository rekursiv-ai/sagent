from collections.abc import Callable

from torch import Tensor

from .descriptors import AOTInput
from .schemas import ViewAndMutationMeta

"""
This module is one of the analysis modules - it takes as input a function or graph
and some preexisting properties, and returns some data that is useful for deciding
how to further proceed with compilation or construct runtime wrappers.

In particular, the analysis here constructs view and mutation metadata from running
a functionalized version of the graph under compilation.
"""
zip = ...
log = ...
static_input_logger = ...

def coerce_tangent_and_suggest_memory_format(
    x: Tensor,
) -> tuple[
    Any | Tensor, list[MemoryFormatMeta | None] | MemoryFormatMeta | None, bool
]: ...
def run_functionalized_fw_and_collect_metadata(
    f,
    *,
    flat_args_descs: list[AOTInput],
    keep_input_mutations: bool,
    is_train: bool = ...,
    static_input_indices: list[int] | None = ...,
    pre_dispatch: bool = ...,
    is_export: bool = ...,
) -> Callable[..., ViewAndMutationMeta]: ...
