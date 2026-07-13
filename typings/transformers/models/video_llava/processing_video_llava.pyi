from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from ...utils import TensorType

"""
Processor class for VideoLlava.
"""
logger = ...

class VideoLlavaProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    video_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor=...,
        video_processor=...,
        tokenizer=...,
        patch_size=...,
        vision_feature_select_strategy=...,
        image_token=...,
        video_token=...,
        chat_template=...,
        num_additional_image_tokens=...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        images: ImageInput | None = ...,
        videos: ImageInput | None = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy = ...,
        max_length=...,
        return_tensors: str | TensorType | None = ...,
    ) -> BatchFeature: ...

__all__ = ["VideoLlavaProcessor"]
