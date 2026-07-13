from transformers.tokenization_utils_base import (
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)

from ...processing_utils import ProcessorMixin
from ...utils import PaddingStrategy, TensorType

"""
Processor class for Nougat.
"""

class NougatProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor, tokenizer) -> None: ...
    def __call__(
        self,
        images=...,
        text=...,
        do_crop_margin: bool | None = ...,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        do_thumbnail: bool | None = ...,
        do_align_long_axis: bool | None = ...,
        do_pad: bool | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        data_format: ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        text_pair: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = ...,
        text_target: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = ...,
        text_pair_target: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy | None = ...,
        max_length: int | None = ...,
        stride: int = ...,
        is_split_into_words: bool = ...,
        pad_to_multiple_of: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_token_type_ids: bool | None = ...,
        return_attention_mask: bool | None = ...,
        return_overflowing_tokens: bool = ...,
        return_special_tokens_mask: bool = ...,
        return_offsets_mapping: bool = ...,
        return_length: bool = ...,
        verbose: bool = ...,
    ): ...
    def post_process_generation(self, *args, **kwargs): ...

__all__ = ["NougatProcessor"]
