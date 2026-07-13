from dataclasses import dataclass

import torch

"""
IMPORTANT NOTICE: Every class and function in this file is deprecated in favor of using the much more general
`masking_utils.py` primitives. New code should not rely on it, it is only kept for backward compatibility for now,
and will be removed in the future.
"""

@dataclass
class AttentionMaskConverter:
    is_causal: bool
    sliding_window: int
    def __init__(self, is_causal: bool, sliding_window: int | None = ...) -> None: ...
    def to_causal_4d(
        self,
        batch_size: int,
        query_length: int,
        key_value_length: int,
        dtype: torch.dtype,
        device: torch.device | str = ...,
    ) -> torch.Tensor | None: ...
    def to_4d(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: int | None = ...,
    ) -> torch.Tensor: ...
