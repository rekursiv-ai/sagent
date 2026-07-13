from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from ...utils import TensorType

"""
Processor class for LayoutLMv2.
"""

class LayoutLMv2Processor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor=..., tokenizer=..., **kwargs) -> None: ...
    def __call__(
        self,
        images,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        text_pair: PreTokenizedInput | list[PreTokenizedInput] | None = ...,
        boxes: list[list[int]] | list[list[list[int]]] | None = ...,
        word_labels: list[int] | list[list[int]] | None = ...,
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
    def get_overflowing_images(
        self, images, overflow_to_sample_mapping
    ):  # -> list[Any]:
        ...
    @property
    def model_input_names(self):  # -> list[str]:
        ...
    @property
    def feature_extractor_class(self):  # -> str:
        ...
    @property
    def feature_extractor(self): ...

__all__ = ["LayoutLMv2Processor"]
