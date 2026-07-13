from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput

"""
Processor class for Blip.
"""

class BlipProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class BlipProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor, tokenizer, **kwargs) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: str | list[str] | TextInput | PreTokenizedInput | None = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[BlipProcessorKwargs],
    ) -> BatchEncoding: ...
    @property
    def model_input_names(self): ...

__all__ = ["BlipProcessor"]
