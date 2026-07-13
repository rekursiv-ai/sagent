from ...file_utils import TensorType
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    TruncationStrategy,
)

"""
Processor class for MarkupLM.
"""

class MarkupLMProcessor(ProcessorMixin):
    feature_extractor_class = ...
    tokenizer_class = ...
    parse_html = ...
    def __call__(
        self,
        html_strings=...,
        nodes=...,
        xpaths=...,
        node_labels=...,
        questions=...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy = ...,
        max_length: int | None = ...,
        stride: int = ...,
        pad_to_multiple_of: int | None = ...,
        return_token_type_ids: bool | None = ...,
        return_attention_mask: bool | None = ...,
        return_overflowing_tokens: bool = ...,
        return_special_tokens_mask: bool = ...,
        return_offsets_mapping: bool = ...,
        return_length: bool = ...,
        verbose: bool = ...,
        return_tensors: str | TensorType | None = ...,
        **kwargs,
    ) -> BatchEncoding: ...

__all__ = ["MarkupLMProcessor"]
