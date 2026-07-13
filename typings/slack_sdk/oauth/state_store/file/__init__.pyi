from logging import Logger

from ..async_state_store import AsyncOAuthStateStore
from ..state_store import OAuthStateStore

class FileOAuthStateStore(OAuthStateStore, AsyncOAuthStateStore):
    def __init__(
        self,
        *,
        expiration_seconds: int,
        base_dir: str = ...,
        client_id: str | None = ...,
        logger: Logger = ...,
    ) -> None: ...
    @property
    def logger(self) -> Logger: ...
    async def async_issue(self, *args, **kwargs) -> str: ...
    async def async_consume(self, state: str) -> bool: ...
    def issue(self, *args, **kwargs) -> str: ...
    def consume(self, state: str) -> bool: ...
