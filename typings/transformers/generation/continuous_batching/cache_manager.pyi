from abc import ABC, abstractmethod
from collections import deque

class CacheAllocator(ABC):
    _index: int
    _block_table: dict[str, list[int]]
    @abstractmethod
    def allocate_blocks(
        self, n_blocks: int, request_id: str, free_blocks: deque[int]
    ) -> int | None: ...
    def free_blocks(self, request_id: str, free_blocks: deque[int]) -> None: ...
    @abstractmethod
    def get_read_indices(
        self, request_id: str, past_length: int, query_length: int
    ) -> list[int]: ...
    @abstractmethod
    def get_write_indices(
        self, request_id: str, past_length: int, query_length: int
    ) -> list[int]: ...
    @abstractmethod
    def get_seqlens_k(
        self, request_id: str, past_length: int, query_length: int
    ) -> tuple[str, int]: ...

class FullAttentionCacheAllocator(CacheAllocator):
    def __init__(self, index: int, block_size: int) -> None: ...
    def allocate_blocks(
        self, n_blocks: int, request_id: str, free_blocks: deque[int]
    ) -> int | None: ...
    def get_read_indices(
        self, request_id: str, past_length: int, query_length: int
    ) -> list[int]: ...
    def get_write_indices(
        self, request_id: str, past_length: int, query_length: int
    ) -> list[int]: ...
    def get_seqlens_k(
        self, request_id: str, past_length: int, query_length: int
    ) -> tuple[str, int]: ...

class SlidingAttentionCacheAllocator(CacheAllocator):
    def __init__(self, index: int, block_size: int, sliding_window: int) -> None: ...
    def allocate_blocks(
        self, n_blocks: int, request_id: str, free_blocks: deque[int]
    ) -> int | None: ...
    def get_read_indices(
        self, request_id: str, past_length: int, query_length: int
    ) -> list[int]: ...
    def get_write_indices(
        self, request_id: str, past_length: int, query_length: int
    ) -> list[int]: ...
    def get_seqlens_k(
        self, request_id: str, past_length: int, query_length: int
    ) -> tuple[str, int]: ...
