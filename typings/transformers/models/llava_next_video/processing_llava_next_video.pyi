from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...video_utils import VideoInput

"""
Processor class for LLaVa-NeXT-Video.
"""
logger = ...

class LlavaNextVideoProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class LlavaNextVideoProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    video_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        video_processor=...,
        image_processor=...,
        tokenizer=...,
        chat_template=...,
        patch_size=...,
        vision_feature_select_strategy=...,
        video_token=...,
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
        videos: VideoInput | None = ...,
        **kwargs: Unpack[LlavaNextVideoProcessorKwargs],
    ) -> BatchFeature: ...

__all__ = ["LlavaNextVideoProcessor"]
