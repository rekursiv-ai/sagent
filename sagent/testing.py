"""Shared test utilities for sagent test mocks."""

from __future__ import annotations

from sagent.custom_types import Pricing


class MockModelCaps:
    """Base capability flags for test model mocks.

    Provides the Model protocol's property/method stubs so individual
    test files only need to add response logic.  Does NOT satisfy the
    full Model protocol alone — concrete mocks must add ``model_id``,
    ``max_request_tokens``, ``buffer``, and ``stream``.
    """

    max_response_tokens: int = 8_192
    supports_streaming: bool = True
    supports_thinking: bool = False
    supports_effort: bool = False
    supports_cache_control: bool = False
    supports_context_management: bool = False
    supports_persistent_retry: bool = False
    supports_account_auth: bool = False
    max_image_dim: int = 8000
    max_image_bytes: int = 5 * 1024 * 1024

    @property
    def pricing(self) -> Pricing:
        return Pricing()

    def estimate_text_token_count(self, text: str) -> int:
        return len(text) // 4

    def estimate_image_token_count(self, data: bytes) -> int:
        del data
        return 256

    def is_context_overflow(self, error: Exception) -> bool:
        del error
        return False
