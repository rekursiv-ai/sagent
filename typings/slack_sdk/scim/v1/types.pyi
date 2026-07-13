from typing import Any

from .default_arg import DefaultArg

class TypeAndValue:
    primary: bool | None | DefaultArg
    type: str | None | DefaultArg
    value: str | None | DefaultArg
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        primary: bool | None | DefaultArg = ...,
        type: str | None | DefaultArg = ...,
        value: str | None | DefaultArg = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict: ...
