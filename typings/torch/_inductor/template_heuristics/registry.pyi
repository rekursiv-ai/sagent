from collections.abc import Iterator
from typing import Any

import contextlib

from .base import TemplateConfigHeuristics

"""
Template heuristic registry system for PyTorch Inductor.

This module provides a centralized registration system for template heuristics,
allowing automatic registration based on device type and conditional registration
for CUDA vs ROCm based on torch.version.hip.
"""
_TEMPLATE_HEURISTIC_REGISTRY: dict[
    tuple[str | None, ...], type[TemplateConfigHeuristics]
] = ...
_HEURISTIC_CACHE: dict[tuple[str, str, str], TemplateConfigHeuristics] = ...
log = ...

def register_template_heuristic(
    template_name: str,
    device_type: str | None,
    register: bool = ...,
    op_name: str | None = ...,
) -> Any: ...
def get_template_heuristic(
    template_name: str, device_type: str, op_name: str
) -> TemplateConfigHeuristics: ...
def clear_registry() -> None: ...
@contextlib.contextmanager
def override_template_heuristics(
    device_type: str, template_op_pairs: list[tuple[str, str]]
) -> Iterator[None]: ...
