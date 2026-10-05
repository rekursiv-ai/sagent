from collections.abc import Callable, Mapping

from .exceptions import (
    JsonSchemaDefinitionException,
    JsonSchemaException,
    JsonSchemaValueException,
)

__all__ = (
    "JsonSchemaDefinitionException",
    "JsonSchemaException",
    "JsonSchemaValueException",
    "compile",
)

def compile(
    definition: Mapping[str, object] | bool,
    handlers: Mapping[str, Callable[[str], object]] = ...,
    formats: Mapping[str, object] = ...,
    use_default: bool = ...,
    use_formats: bool = ...,
    detailed_exceptions: bool = ...,
    fast_fail: bool = ...,
) -> Callable[[object], object]: ...
