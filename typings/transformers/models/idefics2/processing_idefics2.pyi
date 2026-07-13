from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

"""
Processor class for IDEFICS2.
"""
logger = ...

def is_url(val) -> bool: ...
def is_image_or_image_url(elem):  # -> bool:
    ...

class Idefics2ImagesKwargs(ImagesKwargs, total=False):
    image_seq_len: int | None

class Idefics2ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Idefics2ImagesKwargs
    _defaults = ...

class Idefics2Processor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor,
        tokenizer=...,
        image_seq_len: int = ...,
        chat_template: str | None = ...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        images: ImageInput | list[ImageInput] | list[list[ImageInput]] = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[Idefics2ProcessorKwargs],
    ) -> BatchFeature: ...

__all__ = ["Idefics2Processor"]
