from typing import Any

from .default_arg import DefaultArg
from .types import TypeAndValue

class UserAddress:
    country: str | DefaultArg | None
    locality: str | DefaultArg | None
    postal_code: str | DefaultArg | None
    primary: bool | DefaultArg | None
    region: str | DefaultArg | None
    street_address: str | DefaultArg | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        country: str | DefaultArg | None = ...,
        locality: str | DefaultArg | None = ...,
        postal_code: str | DefaultArg | None = ...,
        primary: bool | DefaultArg | None = ...,
        region: str | DefaultArg | None = ...,
        street_address: str | DefaultArg | None = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict: ...

class UserEmail(TypeAndValue): ...
class UserPhoneNumber(TypeAndValue): ...
class UserRole(TypeAndValue): ...

class UserGroup:
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
    def to_dict(self) -> dict: ...

class UserMeta:
    created: str | DefaultArg | None
    location: str | DefaultArg | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        created: str | DefaultArg | None = ...,
        location: str | DefaultArg | None = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict: ...

class UserName:
    family_name: str | DefaultArg | None
    given_name: str | DefaultArg | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        family_name: str | DefaultArg | None = ...,
        given_name: str | DefaultArg | None = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict: ...

class UserPhoto:
    type: str | DefaultArg | None
    value: str | DefaultArg | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        type: str | DefaultArg | None = ...,
        value: str | DefaultArg | None = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict: ...

class User:
    active: bool | DefaultArg | None
    addresses: list[UserAddress] | DefaultArg | None
    display_name: str | DefaultArg | None
    emails: list[TypeAndValue] | DefaultArg | None
    external_id: str | DefaultArg | None
    groups: list[UserGroup] | DefaultArg | None
    id: str | DefaultArg | None
    meta: UserMeta | DefaultArg | None
    name: UserName | DefaultArg | None
    nick_name: str | DefaultArg | None
    phone_numbers: list[TypeAndValue] | DefaultArg | None
    photos: list[UserPhoto] | DefaultArg | None
    profile_url: str | DefaultArg | None
    roles: list[TypeAndValue] | DefaultArg | None
    schemas: list[str] | DefaultArg | None
    timezone: str | DefaultArg | None
    title: str | DefaultArg | None
    user_name: str | DefaultArg | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        active: bool | DefaultArg | None = ...,
        addresses: list[UserAddress | dict[str, Any]] | DefaultArg | None = ...,
        display_name: str | DefaultArg | None = ...,
        emails: list[TypeAndValue | dict[str, Any]] | DefaultArg | None = ...,
        external_id: str | DefaultArg | None = ...,
        groups: list[UserGroup | dict[str, Any]] | DefaultArg | None = ...,
        id: str | DefaultArg | None = ...,
        meta: UserMeta | dict[str, Any] | DefaultArg | None = ...,
        name: UserName | dict[str, Any] | DefaultArg | None = ...,
        nick_name: str | DefaultArg | None = ...,
        phone_numbers: list[TypeAndValue | dict[str, Any]] | DefaultArg | None = ...,
        photos: list[UserPhoto | dict[str, Any]] | DefaultArg | None = ...,
        profile_url: str | DefaultArg | None = ...,
        roles: list[TypeAndValue | dict[str, Any]] | DefaultArg | None = ...,
        schemas: list[str] | DefaultArg | None = ...,
        timezone: str | DefaultArg | None = ...,
        title: str | DefaultArg | None = ...,
        user_name: str | DefaultArg | None = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict[Any, Any]: ...
