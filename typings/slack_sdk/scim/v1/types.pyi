from typing import Any

from .default_arg import DefaultArg

class TypeAndValue:
    primary: bool | DefaultArg | None
    type: str | DefaultArg | None
    value: str | DefaultArg | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        primary: bool | DefaultArg | None = ...,
        type: str | DefaultArg | None = ...,
        value: str | DefaultArg | None = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict: ...
