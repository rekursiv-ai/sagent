from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput

"""
Processor class for Pix2Struct.
"""

class Pix2StructImagesKwargs(ImagesKwargs, total=False):
    max_patches: int | None
    header_text: (
        TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None
    )

class Pix2StructProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Pix2StructImagesKwargs
    _defaults = ...

logger = ...

class Pix2StructProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor, tokenizer) -> None: ...
    def __call__(
        self,
        images=...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[Pix2StructProcessorKwargs],
    ) -> BatchEncoding | BatchFeature: ...
    @property
    def model_input_names(self): ...

__all__ = ["Pix2StructProcessor"]
