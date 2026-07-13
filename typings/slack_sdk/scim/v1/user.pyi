from typing import Any

from .default_arg import DefaultArg
from .types import TypeAndValue

class UserAddress:
    country: str | None | DefaultArg
    locality: str | None | DefaultArg
    postal_code: str | None | DefaultArg
    primary: bool | None | DefaultArg
    region: str | None | DefaultArg
    street_address: str | None | DefaultArg
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        country: str | None | DefaultArg = ...,
        locality: str | None | DefaultArg = ...,
        postal_code: str | None | DefaultArg = ...,
        primary: bool | None | DefaultArg = ...,
        region: str | None | DefaultArg = ...,
        street_address: str | None | DefaultArg = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict: ...

class UserEmail(TypeAndValue): ...
class UserPhoneNumber(TypeAndValue): ...
class UserRole(TypeAndValue): ...

class UserGroup:
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
    def to_dict(self) -> dict: ...

class UserMeta:
    created: str | None | DefaultArg
    location: str | None | DefaultArg
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        created: str | None | DefaultArg = ...,
        location: str | None | DefaultArg = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict: ...

class UserName:
    family_name: str | None | DefaultArg
    given_name: str | None | DefaultArg
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        family_name: str | None | DefaultArg = ...,
        given_name: str | None | DefaultArg = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict: ...

class UserPhoto:
    type: str | None | DefaultArg
    value: str | None | DefaultArg
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        type: str | None | DefaultArg = ...,
        value: str | None | DefaultArg = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict: ...

class User:
    active: bool | None | DefaultArg
    addresses: list[UserAddress] | None | DefaultArg
    display_name: str | None | DefaultArg
    emails: list[TypeAndValue] | None | DefaultArg
    external_id: str | None | DefaultArg
    groups: list[UserGroup] | None | DefaultArg
    id: str | None | DefaultArg
    meta: UserMeta | None | DefaultArg
    name: UserName | None | DefaultArg
    nick_name: str | None | DefaultArg
    phone_numbers: list[TypeAndValue] | None | DefaultArg
    photos: list[UserPhoto] | None | DefaultArg
    profile_url: str | None | DefaultArg
    roles: list[TypeAndValue] | None | DefaultArg
    schemas: list[str] | None | DefaultArg
    timezone: str | None | DefaultArg
    title: str | None | DefaultArg
    user_name: str | None | DefaultArg
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        active: bool | None | DefaultArg = ...,
        addresses: list[UserAddress | dict[str, Any]] | None | DefaultArg = ...,
        display_name: str | None | DefaultArg = ...,
        emails: list[TypeAndValue | dict[str, Any]] | None | DefaultArg = ...,
        external_id: str | None | DefaultArg = ...,
        groups: list[UserGroup | dict[str, Any]] | None | DefaultArg = ...,
        id: str | None | DefaultArg = ...,
        meta: UserMeta | dict[str, Any] | None | DefaultArg = ...,
        name: UserName | dict[str, Any] | None | DefaultArg = ...,
        nick_name: str | None | DefaultArg = ...,
        phone_numbers: list[TypeAndValue | dict[str, Any]] | None | DefaultArg = ...,
        photos: list[UserPhoto | dict[str, Any]] | None | DefaultArg = ...,
        profile_url: str | None | DefaultArg = ...,
        roles: list[TypeAndValue | dict[str, Any]] | None | DefaultArg = ...,
        schemas: list[str] | None | DefaultArg = ...,
        timezone: str | None | DefaultArg = ...,
        title: str | None | DefaultArg = ...,
        user_name: str | None | DefaultArg = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict[Any, Any]: ...
