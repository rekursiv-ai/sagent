from .builtin_handlers import ConnectionErrorRetryHandler, RateLimitErrorRetryHandler
from .builtin_interval_calculators import (
    BackoffRetryIntervalCalculator,
    FixedValueRetryIntervalCalculator,
)
from .handler import RetryHandler
from .interval_calculator import RetryIntervalCalculator
from .jitter import Jitter
from .request import HttpRequest
from .response import HttpResponse
from .state import RetryState

connect_error_retry_handler = ...
rate_limit_error_retry_handler = ...

def default_retry_handlers() -> list[RetryHandler]: ...
def all_builtin_retry_handlers() -> list[RetryHandler]: ...

__all__ = [
    "BackoffRetryIntervalCalculator",
    "ConnectionErrorRetryHandler",
    "FixedValueRetryIntervalCalculator",
    "HttpRequest",
    "HttpResponse",
    "Jitter",
    "RateLimitErrorRetryHandler",
    "RetryHandler",
    "RetryIntervalCalculator",
    "RetryState",
    "all_builtin_retry_handlers",
    "connect_error_retry_handler",
    "default_retry_handlers",
    "rate_limit_error_retry_handler",
]
