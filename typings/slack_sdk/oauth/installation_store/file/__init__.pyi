from logging import Logger

from slack_sdk.oauth.installation_store.async_installation_store import (
    AsyncInstallationStore,
)
from slack_sdk.oauth.installation_store.installation_store import InstallationStore
from slack_sdk.oauth.installation_store.models.bot import Bot
from slack_sdk.oauth.installation_store.models.installation import Installation

class FileInstallationStore(InstallationStore, AsyncInstallationStore):
    def __init__(
        self,
        *,
        base_dir: str = ...,
        historical_data_enabled: bool = ...,
        client_id: str | None = ...,
        logger: Logger = ...,
    ) -> None: ...
    @property
    def logger(self) -> Logger: ...
    async def async_save(self, installation: Installation) -> None: ...
    async def async_save_bot(self, bot: Bot) -> None: ...
    def save(self, installation: Installation) -> None: ...
    def save_bot(self, bot: Bot) -> None: ...
    async def async_find_bot(
        self,
        *,
        enterprise_id: str | None,
        team_id: str | None,
        is_enterprise_install: bool | None = ...,
    ) -> Bot | None: ...
    def find_bot(
        self,
        *,
        enterprise_id: str | None,
        team_id: str | None,
        is_enterprise_install: bool | None = ...,
    ) -> Bot | None: ...
    async def async_find_installation(
        self,
        *,
        enterprise_id: str | None,
        team_id: str | None,
        user_id: str | None = ...,
        is_enterprise_install: bool | None = ...,
    ) -> Installation | None: ...
    def find_installation(
        self,
        *,
        enterprise_id: str | None,
        team_id: str | None,
        user_id: str | None = ...,
        is_enterprise_install: bool | None = ...,
    ) -> Installation | None: ...
    async def async_delete_bot(
        self, *, enterprise_id: str | None, team_id: str | None
    ) -> None: ...
    def delete_bot(self, *, enterprise_id: str | None, team_id: str | None) -> None: ...
    async def async_delete_installation(
        self,
        *,
        enterprise_id: str | None,
        team_id: str | None,
        user_id: str | None = ...,
    ) -> None: ...
    def delete_installation(
        self,
        *,
        enterprise_id: str | None,
        team_id: str | None,
        user_id: str | None = ...,
    ) -> None: ...
