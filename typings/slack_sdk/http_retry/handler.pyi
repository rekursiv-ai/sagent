from slack_sdk.http_retry.interval_calculator import RetryIntervalCalculator
from slack_sdk.http_retry.request import HttpRequest
from slack_sdk.http_retry.response import HttpResponse
from slack_sdk.http_retry.state import RetryState

"""RetryHandler interface.
You can pass an array of handlers to customize retry logics in supported API clients.
"""
default_interval_calculator = ...

class RetryHandler:
    max_retry_count: int
    interval_calculator: RetryIntervalCalculator
    def __init__(
        self,
        max_retry_count: int = ...,
        interval_calculator: RetryIntervalCalculator = ...,
    ) -> None: ...
    def can_retry(
        self,
        *,
        state: RetryState,
        request: HttpRequest,
        response: HttpResponse | None = ...,
        error: Exception | None = ...,
    ) -> bool: ...
    def prepare_for_next_attempt(
        self,
        *,
        state: RetryState,
        request: HttpRequest,
        response: HttpResponse | None = ...,
        error: Exception | None = ...,
    ) -> None: ...
