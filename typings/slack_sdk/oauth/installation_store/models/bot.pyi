from collections.abc import Sequence
from datetime import datetime
from typing import Any

class Bot:
    app_id: str | None
    enterprise_id: str | None
    enterprise_name: str | None
    team_id: str | None
    team_name: str | None
    bot_token: str
    bot_id: str
    bot_user_id: str
    bot_scopes: Sequence[str]
    bot_refresh_token: str | None
    bot_token_expires_at: int | None
    is_enterprise_install: bool
    installed_at: float
    custom_values: dict[str, Any]
    def __init__(
        self,
        *,
        app_id: str | None = ...,
        enterprise_id: str | None = ...,
        enterprise_name: str | None = ...,
        team_id: str | None = ...,
        team_name: str | None = ...,
        bot_token: str,
        bot_id: str,
        bot_user_id: str,
        bot_scopes: str | Sequence[str] = ...,
        bot_refresh_token: str | None = ...,
        bot_token_expires_in: int | None = ...,
        bot_token_expires_at: int | datetime | str | None = ...,
        is_enterprise_install: bool | None = ...,
        installed_at: float | datetime | str,
        custom_values: dict[str, Any] | None = ...,
    ) -> None: ...
    def set_custom_value(self, name: str, value: Any) -> None: ...
    def get_custom_value(self, name: str) -> None: ...
    def to_dict_for_copying(self) -> dict[str, Any]: ...
    def to_dict(self) -> dict[str, Any]: ...
