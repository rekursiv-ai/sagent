from collections.abc import Callable
from typing import Any

from cutlass_library import DataType, EpilogueScheduleType, TileDescription
from sympy import Expr
from torch._inductor.ir import ComputedBuffer, InputBuffer

from ..cutlass_utils import try_import_cutlass

type EpilogueFunctor = Any
type Buffer = ComputedBuffer | InputBuffer
type CutlassTupleType = Any
type CutlassVisitorType = Any
type CutlassArgType = Any
if try_import_cutlass():
    _CUTLASS_C_DTYPES = ...
    class EVTArgRenames:
        def __init__(self) -> None: ...
        def new_name(self, name: str) -> str: ...
        def get(self, name: str) -> str: ...

    def create_example_tensors(
        var_name_to_buffer_name: dict[str, str],
        name_to_buffer: dict[str, Buffer],
        size_hint_fn: Callable[[Expr | int], int],
    ) -> dict[str, python_cutlass.backend.evt.ir.tensor.Tensor]: ...
    def trace(
        fn_src: str,
        example_tensors: dict[str, python_cutlass.backend.evt.ir.tensor.Tensor],
        accum_type: DataType,
        output_type: DataType,
        tile_description: TileDescription,
        epilogue_schedule: EpilogueScheduleType,
        name_to_buffer: dict[str, Buffer],
        size_hint_fn: Callable[[Expr | int], int],
        **kwargs: dict[str, Any],
    ) -> tuple[str, str, str, EVTArgRenames]: ...
