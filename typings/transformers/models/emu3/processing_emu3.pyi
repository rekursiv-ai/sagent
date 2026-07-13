from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput

class Emu3TextKwargs(TextKwargs, total=False):
    return_for_image_generation: bool

class Emu3ImagesKwargs(ImagesKwargs, total=False):
    ratio: str
    image_area: int

class Emu3ProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: Emu3TextKwargs
    images_kwargs: Emu3ImagesKwargs
    _defaults = ...

class Emu3Processor(ProcessorMixin):
    attributes = ...
    tokenizer_class = ...
    image_processor_class = ...
    def __init__(
        self, image_processor, tokenizer, chat_template=..., **kwargs
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
        **kwargs: Unpack[Emu3ProcessorKwargs],
    ) -> BatchFeature: ...
    def calculate_generate_size(
        self, ratio, image_area, spatial_factor
    ):  # -> tuple[int, int]:
        ...
    def postprocess(self, images: ImageInput, **kwargs): ...

__all__ = ["Emu3Processor"]
