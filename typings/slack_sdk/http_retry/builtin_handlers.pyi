from slack_sdk.http_retry.handler import RetryHandler
from slack_sdk.http_retry.interval_calculator import RetryIntervalCalculator
from slack_sdk.http_retry.request import HttpRequest
from slack_sdk.http_retry.response import HttpResponse
from slack_sdk.http_retry.state import RetryState

class ConnectionErrorRetryHandler(RetryHandler):
    def __init__(
        self,
        max_retry_count: int = ...,
        interval_calculator: RetryIntervalCalculator = ...,
        error_types: list[type[Exception]] = ...,
    ) -> None: ...

class RateLimitErrorRetryHandler(RetryHandler):
    def prepare_for_next_attempt(
        self,
        *,
        state: RetryState,
        request: HttpRequest,
        response: HttpResponse | None = ...,
        error: Exception | None = ...,
    ) -> None: ...

class ServerErrorRetryHandler(RetryHandler):
    def __init__(
        self,
        max_retry_count: int = ...,
        interval_calculator: RetryIntervalCalculator = ...,
    ) -> None: ...
