from transformers.processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput

logger = ...

class GotOcr2TextKwargs(TextKwargs, total=False):
    format: bool | None

class GotOcr2ImagesKwargs(ImagesKwargs, total=False):
    box: list | tuple[float, float] | tuple[float, float, float, float] | None
    color: str | None
    num_image_tokens: int | None
    multi_page: bool | None
    crop_to_patches: bool | None
    min_patches: int | None
    max_patches: int | None

class GotOcr2ProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: GotOcr2TextKwargs
    images_kwargs: GotOcr2ImagesKwargs
    _defaults = ...

def preprocess_box_annotation(
    box: list | tuple, image_size: tuple[int, int]
) -> list: ...

class GotOcr2Processor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self, image_processor=..., tokenizer=..., chat_template=..., **kwargs
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
        **kwargs: Unpack[GotOcr2ProcessorKwargs],
    ) -> BatchFeature: ...

__all__ = ["GotOcr2Processor"]
