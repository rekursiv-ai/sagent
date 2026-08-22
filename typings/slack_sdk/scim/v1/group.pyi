from typing import Any

from .default_arg import DefaultArg

class GroupMember:
    display: str | DefaultArg | None
    value: str | DefaultArg | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        display: str | DefaultArg | None = ...,
        value: str | DefaultArg | None = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict[Any, Any]: ...

class GroupMeta:
    created: str | DefaultArg | None
    location: str | DefaultArg | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        created: str | DefaultArg | None = ...,
        location: str | DefaultArg | None = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict[Any, Any]: ...

class Group:
    display_name: str | DefaultArg | None
    id: str | DefaultArg | None
    members: list[GroupMember] | DefaultArg | None
    meta: GroupMeta | DefaultArg | None
    schemas: list[str] | DefaultArg | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        display_name: str | DefaultArg | None = ...,
        id: str | DefaultArg | None = ...,
        members: list[GroupMember] | DefaultArg | None = ...,
        meta: GroupMeta | DefaultArg | None = ...,
        schemas: list[str] | DefaultArg | None = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict[Any, Any]: ...
