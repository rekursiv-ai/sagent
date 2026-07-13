from collections.abc import Sequence
from datetime import datetime
from typing import Any

from slack_sdk.oauth.installation_store.models.bot import Bot

class Installation:
    app_id: str | None
    enterprise_id: str | None
    enterprise_name: str | None
    enterprise_url: str | None
    team_id: str | None
    team_name: str | None
    bot_token: str | None
    bot_id: str | None
    bot_user_id: str | None
    bot_scopes: Sequence[str] | None
    bot_refresh_token: str | None
    bot_token_expires_at: int | None
    user_id: str
    user_token: str | None
    user_scopes: Sequence[str] | None
    user_refresh_token: str | None
    user_token_expires_at: int | None
    incoming_webhook_url: str | None
    incoming_webhook_channel: str | None
    incoming_webhook_channel_id: str | None
    incoming_webhook_configuration_url: str | None
    is_enterprise_install: bool
    token_type: str | None
    installed_at: float
    custom_values: dict[str, Any]
    def __init__(
        self,
        *,
        app_id: str | None = ...,
        enterprise_id: str | None = ...,
        enterprise_name: str | None = ...,
        enterprise_url: str | None = ...,
        team_id: str | None = ...,
        team_name: str | None = ...,
        bot_token: str | None = ...,
        bot_id: str | None = ...,
        bot_user_id: str | None = ...,
        bot_scopes: str | Sequence[str] = ...,
        bot_refresh_token: str | None = ...,
        bot_token_expires_in: int | None = ...,
        bot_token_expires_at: int | datetime | str | None = ...,
        user_id: str,
        user_token: str | None = ...,
        user_scopes: str | Sequence[str] = ...,
        user_refresh_token: str | None = ...,
        user_token_expires_in: int | None = ...,
        user_token_expires_at: int | datetime | str | None = ...,
        incoming_webhook_url: str | None = ...,
        incoming_webhook_channel: str | None = ...,
        incoming_webhook_channel_id: str | None = ...,
        incoming_webhook_configuration_url: str | None = ...,
        is_enterprise_install: bool | None = ...,
        token_type: str | None = ...,
        installed_at: float | datetime | str | None = ...,
        custom_values: dict[str, Any] | None = ...,
    ) -> None: ...
    def to_bot(self) -> Bot: ...
    def set_custom_value(self, name: str, value: Any) -> None: ...
    def get_custom_value(self, name: str) -> Any | None: ...
    def to_dict_for_copying(self) -> dict[str, Any]: ...
    def to_dict(self) -> dict[str, Any]: ...
