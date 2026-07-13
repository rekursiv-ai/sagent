from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...video_utils import VideoInput

"""
Processor class for Qwen2-VL.
"""
logger = ...

class Qwen2VLImagesKwargs(ImagesKwargs):
    min_pixels: int | None
    max_pixels: int | None
    patch_size: int | None
    temporal_patch_size: int | None
    merge_size: int | None

class Qwen2VLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Qwen2VLImagesKwargs
    _defaults = ...

class Qwen2VLProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    video_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor=...,
        tokenizer=...,
        video_processor=...,
        chat_template=...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        videos: VideoInput | None = ...,
        **kwargs: Unpack[Qwen2VLProcessorKwargs],
    ) -> BatchFeature: ...
    def post_process_image_text_to_text(
        self,
        generated_outputs,
        skip_special_tokens=...,
        clean_up_tokenization_spaces=...,
        **kwargs,
    ): ...

__all__ = ["Qwen2VLProcessor"]
