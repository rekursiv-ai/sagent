from contextlib import contextmanager

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

"""
Processor class for TrOCR.
"""

class TrOCRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class TrOCRProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor=..., tokenizer=..., **kwargs) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[TrOCRProcessorKwargs],
    ) -> BatchFeature: ...
    @property
    def model_input_names(self): ...
    @contextmanager
    def as_target_processor(self):  # -> Generator[None, Any, None]:
        ...
    @property
    def feature_extractor_class(self):  # -> str:
        ...
    @property
    def feature_extractor(self): ...

__all__ = ["TrOCRProcessor"]
