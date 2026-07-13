from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...video_utils import VideoInput

"""
Processor class for LLaVa-Onevision.
"""
logger = ...

class LlavaOnevisionProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class LlavaOnevisionProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    video_processor_class = ...
    def __init__(
        self,
        image_processor=...,
        tokenizer=...,
        video_processor=...,
        num_image_tokens=...,
        vision_feature_select_strategy=...,
        chat_template=...,
        image_token=...,
        video_token=...,
        vision_aspect_ratio=...,
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
        videos: VideoInput | None = ...,
        **kwargs: Unpack[LlavaOnevisionProcessorKwargs],
    ) -> BatchFeature: ...

__all__ = ["LlavaOnevisionProcessor"]
