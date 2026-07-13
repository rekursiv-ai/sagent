from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
)
from ...tokenization_utils_base import TextInput

"""
Processor class for Kosmos2_5.
"""

class Kosmos2_5ImagesKwargs(ImagesKwargs, total=False):
    max_patches: int | None
    num_image_tokens: int | None

class Kosmos2_5ProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: TextKwargs
    images_kwargs: Kosmos2_5ImagesKwargs
    _defaults = ...

class Kosmos2_5Processor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor, tokenizer) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput | list[TextInput] = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[Kosmos2_5ProcessorKwargs],
    ) -> BatchFeature: ...
    def batch_decode(self, *args, **kwargs): ...
    def decode(self, *args, **kwargs): ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...

__all__ = ["Kosmos2_5Processor"]
