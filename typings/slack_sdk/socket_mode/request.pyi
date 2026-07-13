from typing import Any

from slack_sdk.models import JsonObject

class SocketModeRequest:
    type: str
    envelope_id: str
    payload: dict[str, Any]
    accepts_response_payload: bool
    retry_attempt: int | None
    retry_reason: str | None
    def __init__(
        self,
        type: str,
        envelope_id: str,
        payload: dict[str, Any] | JsonObject | str,
        accepts_response_payload: bool | None = ...,
        retry_attempt: int | None = ...,
        retry_reason: str | None = ...,
    ) -> None: ...
    @classmethod
    def from_dict(cls, message: dict[str, Any]) -> SocketModeRequest | None: ...
    def to_dict(self) -> dict[str, Any]: ...
