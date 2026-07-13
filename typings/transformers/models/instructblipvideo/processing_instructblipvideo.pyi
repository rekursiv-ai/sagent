from ...image_processing_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from ...utils import TensorType
from ...video_utils import VideoInput

"""
Processor class for InstructBLIP. Largely copy of Blip2Processor with addition of a tokenizer for the Q-Former.
"""
logger = ...

class InstructBlipVideoProcessor(ProcessorMixin):
    attributes = ...
    video_processor_class = ...
    tokenizer_class = ...
    qformer_tokenizer_class = ...
    def __init__(
        self,
        video_processor,
        tokenizer,
        qformer_tokenizer,
        num_query_tokens=...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        images: VideoInput | None = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy = ...,
        max_length: int | None = ...,
        stride: int = ...,
        pad_to_multiple_of: int | None = ...,
        return_attention_mask: bool | None = ...,
        return_overflowing_tokens: bool = ...,
        return_special_tokens_mask: bool = ...,
        return_offsets_mapping: bool = ...,
        return_token_type_ids: bool = ...,
        return_length: bool = ...,
        verbose: bool = ...,
        return_tensors: str | TensorType | None = ...,
        **kwargs,
    ) -> BatchFeature: ...
    @property
    def model_input_names(self): ...
    def save_pretrained(self, save_directory, **kwargs):  # -> list[str]:
        ...
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, **kwargs
    ):  # -> InstructBlipVideoProcessor:
        ...

__all__ = ["InstructBlipVideoProcessor"]
