from abc import ABC, abstractmethod

from .cache import PagedAttentionCache
from .requests import RequestState
from ...utils.metrics import attach_tracer, traced

class Scheduler(ABC):
    def __init__(
        self, cache: PagedAttentionCache, retain_cache_on_finish: bool = ...
    ) -> None: ...
    @traced
    def add_waiting_request(self, state: RequestState):  # -> None:
        ...
    @abstractmethod
    def schedule_batch(self, token_budget: int) -> list[RequestState]: ...
    @traced
    def has_pending_requests(self) -> bool: ...
    @traced
    def finish_request(self, request_id: str, evict_from_cache: bool = ...):  # -> None:
        ...
    @traced
    def get_active_request_static_outputs(self, request_id: str) -> list[int]: ...
    @traced
    def set_request_cancellation(self, request_id: str):  # -> None:
        ...
    @traced
    def clear_cancelled_requests(self):  # -> None:
        ...
    @traced
    def request_is_cancelled(self, request_id: str) -> bool: ...

@attach_tracer()
class FIFOScheduler(Scheduler):
    def __init__(
        self,
        cache: PagedAttentionCache,
        retain_cache_on_finish: bool = ...,
        safety_margin: float = ...,
    ) -> None: ...
    @traced
    def schedule_batch(self, token_budget: int) -> list[RequestState]: ...

@attach_tracer()
class PrefillFirstScheduler(Scheduler):
    @traced
    def schedule_batch(self, token_budget: int) -> list[RequestState]: ...

SCHEDULER_MAPPING = ...
