from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

"""
Processor class for Llava.
"""
logger = ...

class LlavaProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class LlavaProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor=...,
        tokenizer=...,
        patch_size=...,
        vision_feature_select_strategy=...,
        chat_template=...,
        image_token=...,
        num_additional_image_tokens=...,
        **kwargs,
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
        **kwargs: Unpack[LlavaProcessorKwargs],
    ) -> BatchFeature: ...

__all__ = ["LlavaProcessor"]
