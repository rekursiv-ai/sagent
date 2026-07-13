from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput

"""
Processor class for BLIP-2.
"""
logger = ...

class Blip2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class Blip2Processor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self, image_processor, tokenizer, num_query_tokens=..., **kwargs
    ) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: str | list[str] | TextInput | PreTokenizedInput | None = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[Blip2ProcessorKwargs],
    ) -> BatchEncoding: ...

__all__ = ["Blip2Processor"]
