from collections.abc import Sequence
from typing import Any

"""Compatibility functions for the torch.onnx.export API."""

def from_dynamic_axes_to_dynamic_shapes(
    model,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    *,
    dynamic_axes=...,
    output_names: set[str],
    input_names: Sequence[str] | None = ...,
) -> tuple[dict[str, Any | None] | None, tuple[Any, ...], dict[str, Any] | None]: ...
def from_dynamic_shapes_to_dynamic_axes(
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any],
    input_names: Sequence[str],
    exception: Exception,
) -> dict[str, Any] | None: ...
def convert_str_to_export_dim(
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None,
) -> tuple[dict[str, Any] | tuple[Any, ...] | list[Any] | None, bool]: ...
def create_rename_mapping(
    inputs, dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any]
) -> dict[str, str]: ...
