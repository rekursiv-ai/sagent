from typing import Any
from urllib.request import Request

class HttpRequest:
    method: str
    url: str
    headers: dict[str, str | list[str]]
    body_params: dict[str, Any] | None
    data: bytes | None
    def __init__(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str | list[str]],
        body_params: dict[str, Any] | None = ...,
        data: bytes | None = ...,
    ) -> None: ...
    @classmethod
    def from_urllib_http_request(cls, req: Request) -> HttpRequest: ...
