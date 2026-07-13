from typing import Any

from slack_sdk.scim.v1.group import Group
from slack_sdk.scim.v1.user import User

class Errors:
    code: int
    description: str
    def __init__(self, code: int, description: str) -> None: ...
    def to_dict(self) -> dict: ...

class SCIMResponse:
    url: str
    status_code: int
    headers: dict[str, Any]
    raw_body: str | None
    body: dict[str, Any] | None
    snake_cased_body: dict[str, Any] | None
    errors: Errors | None
    @property
    def snake_cased_body(self) -> dict[str, Any] | None: ...
    @property
    def errors(self) -> Errors | None: ...
    def __init__(
        self, *, url: str, status_code: int, raw_body: str | None, headers: dict
    ) -> None: ...

class SearchUsersResponse(SCIMResponse):
    users: list[User]
    @property
    def users(self) -> list[User]: ...
    def __init__(self, underlying: SCIMResponse) -> None: ...

class ReadUserResponse(SCIMResponse):
    user: User
    @property
    def user(self) -> User: ...
    def __init__(self, underlying: SCIMResponse) -> None: ...

class UserCreateResponse(SCIMResponse):
    user: User
    @property
    def user(self) -> User: ...
    def __init__(self, underlying: SCIMResponse) -> None: ...

class UserPatchResponse(SCIMResponse):
    user: User
    @property
    def user(self) -> User: ...
    def __init__(self, underlying: SCIMResponse) -> None: ...

class UserUpdateResponse(SCIMResponse):
    user: User
    @property
    def user(self) -> User: ...
    def __init__(self, underlying: SCIMResponse) -> None: ...

class UserDeleteResponse(SCIMResponse):
    def __init__(self, underlying: SCIMResponse) -> None: ...

class SearchGroupsResponse(SCIMResponse):
    groups: list[Group]
    @property
    def groups(self) -> list[Group]: ...
    def __init__(self, underlying: SCIMResponse) -> None: ...

class ReadGroupResponse(SCIMResponse):
    group: Group
    @property
    def group(self) -> Group: ...
    def __init__(self, underlying: SCIMResponse) -> None: ...

class GroupCreateResponse(SCIMResponse):
    group: Group
    @property
    def group(self) -> Group: ...
    def __init__(self, underlying: SCIMResponse) -> None: ...

class GroupPatchResponse(SCIMResponse):
    def __init__(self, underlying: SCIMResponse) -> None: ...

class GroupUpdateResponse(SCIMResponse):
    group: Group
    @property
    def group(self) -> Group: ...
    def __init__(self, underlying: SCIMResponse) -> None: ...

class GroupDeleteResponse(SCIMResponse):
    def __init__(self, underlying: SCIMResponse) -> None: ...
