from transformers.processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput

class Llama4ImagesKwargs(ImagesKwargs, total=False):
    max_patches: int | None
    resize_to_max_canvas: bool | None

class Llama4ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Llama4ImagesKwargs
    _defaults = ...

chat_template = ...

class Llama4Processor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor=...,
        tokenizer=...,
        patch_size: int = ...,
        pixel_shuffle_ratio: float = ...,
        fake_image_token=...,
        image_token=...,
        start_of_image_token=...,
        end_of_image_token=...,
        patch_token=...,
        tile_x_separator_token=...,
        tile_y_separator_token=...,
        chat_template=...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[Llama4ProcessorKwargs],
    ) -> BatchFeature: ...

__all__ = ["Llama4Processor"]
