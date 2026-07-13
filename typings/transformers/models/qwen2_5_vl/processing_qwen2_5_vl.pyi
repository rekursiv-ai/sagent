from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
    VideosKwargs,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...video_utils import VideoInput

class Qwen2_5_VLVideosProcessorKwargs(VideosKwargs, total=False):
    fps: list[float] | float

class Qwen2_5_VLImagesKwargs(ImagesKwargs):
    min_pixels: int | None
    max_pixels: int | None
    patch_size: int | None
    temporal_patch_size: int | None
    merge_size: int | None

class Qwen2_5_VLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Qwen2_5_VLImagesKwargs
    videos_kwargs: Qwen2_5_VLVideosProcessorKwargs
    _defaults = ...

class Qwen2_5_VLProcessor(ProcessorMixin):
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
        **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
    ) -> BatchFeature: ...
    def post_process_image_text_to_text(
        self,
        generated_outputs,
        skip_special_tokens=...,
        clean_up_tokenization_spaces=...,
        **kwargs,
    ): ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...

__all__ = ["Qwen2_5_VLProcessor"]
