from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput

"""
Processor class for PaliGemma.
"""
logger = ...
IMAGE_TOKEN = ...
EXTRA_TOKENS = ...

class PaliGemmaTextKwargs(TextKwargs):
    suffix: (
        TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None
    )

class PaliGemmaImagesKwargs(ImagesKwargs):
    do_convert_rgb: bool | None

class PaliGemmaProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: PaliGemmaTextKwargs
    images_kwargs: PaliGemmaImagesKwargs
    _defaults = ...

def is_url(val) -> bool: ...
def is_image_or_image_url(elem):  # -> bool:
    ...
def build_string_from_input(
    prompt, bos_token, image_seq_len, image_token, num_images
):  # -> str:
    ...

class PaliGemmaProcessor(ProcessorMixin):
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
        | list[PreTokenizedInput] = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[PaliGemmaProcessorKwargs],
    ) -> BatchFeature: ...

__all__ = ["PaliGemmaProcessor"]
