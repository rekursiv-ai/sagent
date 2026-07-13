from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

"""
Processor class for Chameleon.
"""

class ChameleonTextKwargs(TextKwargs, total=False):
    return_for_text_completion: bool

class ChameleonProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: ChameleonTextKwargs
    _defaults = ...

class ChameleonProcessor(ProcessorMixin):
    attributes = ...
    tokenizer_class = ...
    image_processor_class = ...
    def __init__(
        self,
        image_processor,
        tokenizer,
        image_seq_length: int = ...,
        image_token: str = ...,
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
        **kwargs: Unpack[ChameleonProcessorKwargs],
    ) -> BatchFeature: ...

__all__ = ["ChameleonProcessor"]
