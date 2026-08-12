from typing import Self

class LegacySlackResponse:
    def __init__(
        self,
        *,
        client,
        http_verb: str,
        api_url: str,
        req_args: dict,
        data: dict | bytes,
        headers: dict,
        status_code: int,
        use_sync_aiohttp: bool = ...,
    ) -> None: ...
    def __getitem__(self, key) -> None: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> Self: ...
    def get(self, key, default=...) -> None: ...
    def validate(self) -> Self: ...
