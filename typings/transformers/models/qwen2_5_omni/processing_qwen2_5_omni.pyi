import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
    VideosKwargs,
)
from ...tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput
from ...video_utils import VideoInput

"""
Processor class for Qwen2.5Omni.
"""

class Qwen2_5_OmniVideosKwargs(VideosKwargs):
    fps: list[int | float] | None
    use_audio_in_video: bool | None
    seconds_per_chunk: float | None
    position_id_per_seconds: int | None
    min_pixels: int | None
    max_pixels: int | None
    patch_size: int | None
    temporal_patch_size: int | None
    merge_size: int | None

class Qwen2_5_OmniImagesKwargs(ImagesKwargs):
    min_pixels: int | None
    max_pixels: int | None
    patch_size: int | None
    temporal_patch_size: int | None
    merge_size: int | None

class Qwen2_5OmniProcessorKwargs(ProcessingKwargs, total=False):
    videos_kwargs: Qwen2_5_OmniVideosKwargs
    images_kwargs: Qwen2_5_OmniImagesKwargs
    _defaults = ...

class Qwen2_5OmniProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    video_processor_class = ...
    feature_extractor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor=...,
        video_processor=...,
        feature_extractor=...,
        tokenizer=...,
        chat_template=...,
    ) -> None: ...
    def __call__(
        self,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        images: ImageInput | None = ...,
        videos: VideoInput | None = ...,
        audio: AudioInput | None = ...,
        **kwargs: Unpack[Qwen2_5OmniProcessorKwargs],
    ) -> BatchFeature: ...
    def replace_multimodal_special_tokens(
        self,
        text,
        audio_lengths,
        image_grid_thw,
        video_grid_thw,
        video_second_per_grid,
        use_audio_in_video,
        position_id_per_seconds,
        seconds_per_chunk,
    ):  # -> list[Any]:
        ...
    def get_chunked_index(
        self, token_indices: np.ndarray, tokens_per_chunk: int
    ) -> list[tuple[int, int]]: ...
    def apply_chat_template(
        self, conversations, chat_template=..., **kwargs
    ):  # -> str:
        ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...

__all__ = ["Qwen2_5OmniProcessor"]
