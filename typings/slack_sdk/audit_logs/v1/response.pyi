from typing import Any

from slack_sdk.audit_logs.v1.logs import LogsResponse

class AuditLogsResponse:
    url: str
    status_code: int
    headers: dict[str, Any]
    raw_body: str | None
    body: dict[str, Any] | None
    typed_body: LogsResponse | None
    @property
    def typed_body(self) -> LogsResponse | None: ...
    def __init__(
        self, *, url: str, status_code: int, raw_body: str | None, headers: dict
    ) -> None: ...
