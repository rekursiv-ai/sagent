from slack_sdk.http_retry.async_handler import AsyncRetryHandler
from slack_sdk.http_retry.interval_calculator import RetryIntervalCalculator
from slack_sdk.http_retry.request import HttpRequest
from slack_sdk.http_retry.response import HttpResponse
from slack_sdk.http_retry.state import RetryState

class AsyncConnectionErrorRetryHandler(AsyncRetryHandler):
    def __init__(
        self,
        max_retry_count: int = ...,
        interval_calculator: RetryIntervalCalculator = ...,
        error_types: list[type[Exception]] = ...,
    ) -> None: ...

class AsyncRateLimitErrorRetryHandler(AsyncRetryHandler):
    async def prepare_for_next_attempt_async(
        self,
        *,
        state: RetryState,
        request: HttpRequest,
        response: HttpResponse | None = ...,
        error: Exception | None = ...,
    ) -> None: ...

class AsyncServerErrorRetryHandler(AsyncRetryHandler):
    def __init__(
        self,
        max_retry_count: int = ...,
        interval_calculator: RetryIntervalCalculator = ...,
    ) -> None: ...

def async_default_handlers() -> list[AsyncRetryHandler]: ...
