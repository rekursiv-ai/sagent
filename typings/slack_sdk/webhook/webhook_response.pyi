from typing import Any

class WebhookResponse:
    def __init__(
        self, *, url: str, status_code: int, body: str, headers: dict[str, Any]
    ) -> None: ...
