from typing import Any

class HttpResponse:
    status_code: int
    headers: dict[str, list[str] | str]
    body: dict[str, Any] | None
    data: bytes | None
    def __init__(
        self,
        *,
        status_code: int | str,
        headers: dict[str, str | list[str]],
        body: dict[str, Any] | None = ...,
        data: bytes | None = ...,
    ) -> None: ...
