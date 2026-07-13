from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...video_utils import VideoInput

"""
Processor class for PerceptionLM.
"""
logger = ...

class PerceptionLMProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class PerceptionLMProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    video_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        video_processor=...,
        image_processor=...,
        tokenizer=...,
        patch_size=...,
        chat_template=...,
        pooling_ratio=...,
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
        **kwargs: Unpack[PerceptionLMProcessorKwargs],
    ) -> BatchFeature: ...

__all__ = ["PerceptionLMProcessor"]
