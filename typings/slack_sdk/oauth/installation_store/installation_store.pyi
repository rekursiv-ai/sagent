from logging import Logger

from .models.bot import Bot
from .models.installation import Installation

"""Slack installation data store

Refer to https://docs.slack.dev/tools/python-slack-sdk/oauth for details.
"""

class InstallationStore:
    @property
    def logger(self) -> Logger: ...
    def save(self, installation: Installation): ...
    def save_bot(self, bot: Bot): ...
    def find_bot(
        self,
        *,
        enterprise_id: str | None,
        team_id: str | None,
        is_enterprise_install: bool | None = ...,
    ) -> Bot | None: ...
    def find_installation(
        self,
        *,
        enterprise_id: str | None,
        team_id: str | None,
        user_id: str | None = ...,
        is_enterprise_install: bool | None = ...,
    ) -> Installation | None: ...
    def delete_bot(self, *, enterprise_id: str | None, team_id: str | None) -> None: ...
    def delete_installation(
        self,
        *,
        enterprise_id: str | None,
        team_id: str | None,
        user_id: str | None = ...,
    ) -> None: ...
    def delete_all(self, *, enterprise_id: str | None, team_id: str | None) -> None: ...
