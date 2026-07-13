from typing import Any

from .default_arg import DefaultArg

class GroupMember:
    display: str | None | DefaultArg
    value: str | None | DefaultArg
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        display: str | None | DefaultArg = ...,
        value: str | None | DefaultArg = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict[Any, Any]: ...

class GroupMeta:
    created: str | None | DefaultArg
    location: str | None | DefaultArg
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        created: str | None | DefaultArg = ...,
        location: str | None | DefaultArg = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict[Any, Any]: ...

class Group:
    display_name: str | None | DefaultArg
    id: str | None | DefaultArg
    members: list[GroupMember] | None | DefaultArg
    meta: GroupMeta | None | DefaultArg
    schemas: list[str] | None | DefaultArg
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        display_name: str | None | DefaultArg = ...,
        id: str | None | DefaultArg = ...,
        members: list[GroupMember] | None | DefaultArg = ...,
        meta: GroupMeta | None | DefaultArg = ...,
        schemas: list[str] | None | DefaultArg = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict[Any, Any]: ...
