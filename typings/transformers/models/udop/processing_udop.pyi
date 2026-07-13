from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

"""
Processor class for UDOP.
"""
logger = ...

class UdopTextKwargs(TextKwargs, total=False):
    word_labels: list[int] | list[list[int]] | None
    boxes: list[list[int]] | list[list[list[int]]]

class UdopProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: UdopTextKwargs
    _defaults = ...

class UdopProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor, tokenizer) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[UdopProcessorKwargs],
    ) -> BatchFeature: ...
    def get_overflowing_images(
        self, images, overflow_to_sample_mapping
    ):  # -> list[Any]:
        ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...

__all__ = ["UdopProcessor"]
