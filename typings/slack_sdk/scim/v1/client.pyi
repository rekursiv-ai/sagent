from ssl import SSLContext
from typing import Any

import logging

from slack_sdk.http_retry.handler import RetryHandler

from .group import Group
from .response import (
    GroupCreateResponse,
    GroupDeleteResponse,
    GroupPatchResponse,
    GroupUpdateResponse,
    ReadGroupResponse,
    ReadUserResponse,
    SCIMResponse,
    SearchGroupsResponse,
    SearchUsersResponse,
    UserCreateResponse,
    UserDeleteResponse,
    UserPatchResponse,
    UserUpdateResponse,
)
from .user import User

"""SCIM API is a set of APIs for provisioning and managing user accounts and groups.
SCIM is used by Single Sign-On (SSO) services and identity providers to manage people across a variety of tools,
including Slack.

Refer to https://docs.slack.dev/tools/python-slack-sdk/scim/ for details.
"""

class SCIMClient:
    BASE_URL = ...
    token: str
    timeout: int
    ssl: SSLContext | None
    proxy: str | None
    base_url: str
    default_headers: dict[str, str]
    logger: logging.Logger
    retry_handlers: list[RetryHandler]
    def __init__(
        self,
        token: str,
        timeout: int = ...,
        ssl: SSLContext | None = ...,
        proxy: str | None = ...,
        base_url: str = ...,
        default_headers: dict[str, str] | None = ...,
        user_agent_prefix: str | None = ...,
        user_agent_suffix: str | None = ...,
        logger: logging.Logger | None = ...,
        retry_handlers: list[RetryHandler] | None = ...,
    ) -> None: ...
    def search_users(
        self, *, count: int, start_index: int, filter: str | None = ...
    ) -> SearchUsersResponse: ...
    def read_user(self, id: str) -> ReadUserResponse: ...
    def create_user(self, user: dict[str, Any] | User) -> UserCreateResponse: ...
    def patch_user(
        self, id: str, partial_user: dict[str, Any] | User
    ) -> UserPatchResponse: ...
    def update_user(self, user: dict[str, Any] | User) -> UserUpdateResponse: ...
    def delete_user(self, id: str) -> UserDeleteResponse: ...
    def search_groups(
        self, *, count: int, start_index: int, filter: str | None = ...
    ) -> SearchGroupsResponse: ...
    def read_group(self, id: str) -> ReadGroupResponse: ...
    def create_group(self, group: dict[str, Any] | Group) -> GroupCreateResponse: ...
    def patch_group(
        self, id: str, partial_group: dict[str, Any] | Group
    ) -> GroupPatchResponse: ...
    def update_group(self, group: dict[str, Any] | Group) -> GroupUpdateResponse: ...
    def delete_group(self, id: str) -> GroupDeleteResponse: ...
    def api_call(
        self,
        *,
        http_verb: str,
        path: str,
        query_params: dict[str, Any] | None = ...,
        body_params: dict[str, Any] | None = ...,
        headers: dict[str, str] | None = ...,
    ) -> SCIMResponse: ...
