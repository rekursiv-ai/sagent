from logging import Logger

from .models.bot import Bot
from .models.installation import Installation

class AsyncInstallationStore:
    @property
    def logger(self) -> Logger: ...
    async def async_save(self, installation: Installation): ...
    async def async_save_bot(self, bot: Bot): ...
    async def async_find_bot(
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
    async def async_delete_bot(
        self, *, enterprise_id: str | None, team_id: str | None
    ) -> None: ...
    async def async_delete_installation(
        self,
        *,
        enterprise_id: str | None,
        team_id: str | None,
        user_id: str | None = ...,
    ) -> None: ...
    async def async_delete_all(
        self, *, enterprise_id: str | None, team_id: str | None
    ) -> None: ...
