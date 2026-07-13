from dataclasses import dataclass
from typing import Any, TypedDict, TypeVar

import os
import typing

import numpy as np

from .audio_utils import AudioInput
from .image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
)
from .tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from .utils import PushToHubMixin, TensorType
from .utils.deprecation import deprecate_kwarg
from .video_utils import VideoInput, VideoMetadata

"""
Processing saving/loading class for common processors.
"""
logger = ...
SpecificProcessorType = TypeVar("SpecificProcessorType", bound=ProcessorMixin)
transformers_module = ...
AUTO_TO_BASE_CLASS_MAPPING = ...
Unpack = typing.Unpack

class TextKwargs(TypedDict, total=False):
    text_pair: (
        TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None
    )
    text_target: (
        TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput]
    )
    text_pair_target: (
        TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None
    )
    add_special_tokens: bool | None
    padding: bool | str | PaddingStrategy
    truncation: bool | str | TruncationStrategy
    max_length: int | None
    stride: int | None
    is_split_into_words: bool | None
    pad_to_multiple_of: int | None
    return_token_type_ids: bool | None
    return_attention_mask: bool | None
    return_overflowing_tokens: bool | None
    return_special_tokens_mask: bool | None
    return_offsets_mapping: bool | None
    return_length: bool | None
    verbose: bool | None
    padding_side: str | None
    return_mm_token_type_ids: bool | None

class ImagesKwargs(TypedDict, total=False):
    do_resize: bool | None
    size: dict[str, int] | None
    crop_size: dict[str, int] | None
    resample: PILImageResampling | int | None
    do_rescale: bool | None
    rescale_factor: float | None
    do_normalize: bool | None
    image_mean: float | list[float] | None
    image_std: float | list[float] | None
    do_pad: bool | None
    pad_size: dict[str, int] | None
    do_center_crop: bool | None
    data_format: ChannelDimension | None
    input_data_format: str | ChannelDimension | None
    device: str | None

class VideosKwargs(TypedDict, total=False):
    do_convert_rgb: bool | None
    do_resize: bool | None
    size: dict[str, int] | None
    default_to_square: bool | None
    resample: PILImageResampling | None
    do_rescale: bool | None
    rescale_factor: float | None
    do_normalize: bool | None
    image_mean: float | list[float] | None
    image_std: float | list[float] | None
    do_center_crop: bool | None
    crop_size: dict[str, int] | None
    data_format: ChannelDimension | None
    input_data_format: str | ChannelDimension | None
    device: str | None
    do_sample_frames: bool | None
    video_metadata: VideoMetadata | dict | None
    fps: int | float | None
    num_frames: int | None
    return_metadata: bool | None

class AudioKwargs(TypedDict, total=False):
    sampling_rate: int | None
    raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]] | None
    padding: bool | str | PaddingStrategy | None
    max_length: int | None
    truncation: bool | None
    pad_to_multiple_of: int | None
    return_attention_mask: bool | None

class CommonKwargs(TypedDict, total=False):
    return_tensors: str | TensorType | None

class ProcessingKwargs(TypedDict, total=False):
    _defaults = ...
    common_kwargs: CommonKwargs = ...
    text_kwargs: TextKwargs = ...
    images_kwargs: ImagesKwargs = ...
    videos_kwargs: VideosKwargs = ...
    audio_kwargs: AudioKwargs = ...

class TokenizerChatTemplateKwargs(TypedDict, total=False):
    tools: list[dict] | None = ...
    documents: list[dict[str, str]] | None = ...
    add_generation_prompt: bool | None = ...
    continue_final_message: bool | None = ...
    return_assistant_tokens_mask: bool | None = ...

class ChatTemplateLoadKwargs(TypedDict, total=False):
    sampling_rate: int | None = ...
    load_audio_from_video: bool | None = ...

class ProcessorChatTemplateKwargs(
    ChatTemplateLoadKwargs, TokenizerChatTemplateKwargs, total=False
):
    tokenize: bool | None = ...
    return_dict: bool | None = ...

class AllKwargsForChatTemplate(TypedDict, total=False):
    processor_kwargs: ProcessingKwargs
    mm_load_kwargs: ChatTemplateLoadKwargs
    template_kwargs: ProcessorChatTemplateKwargs

@dataclass
class MultiModalData:
    num_image_tokens: list[int] | None = ...
    num_video_tokens: list[int] | None = ...
    num_audio_tokens: list[int] | None = ...
    num_image_patches: list[int] | None = ...
    def __contains__(self, key):  # -> bool:
        ...
    def __getitem__(self, key):  # -> Any:
        ...

class ProcessorMixin(PushToHubMixin):
    attributes = ...
    optional_attributes = ...
    optional_call_args: list[str] = ...
    feature_extractor_class = ...
    tokenizer_class = ...
    _auto_class = ...
    valid_processor_kwargs = ProcessingKwargs
    def __init__(self, *args, **kwargs) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = ...,
        videos: VideoInput | None = ...,
        audio: AudioInput | None = ...,
        **kwargs: Unpack[ProcessingKwargs],
    ):  # -> BatchFeature:
        ...
    def check_argument_for_proper_class(
        self, argument_name, argument
    ):  # -> tuple[Any, ...] | Any:
        ...
    def to_dict(self, legacy_serialization=...) -> dict[str, Any]: ...
    def to_json_string(self, legacy_serialization=...) -> str: ...
    def to_json_file(
        self, json_file_path: str | os.PathLike, legacy_serialization=...
    ):  # -> None:
        ...
    def __repr__(self):  # -> str:
        ...
    def save_pretrained(
        self,
        save_directory,
        push_to_hub: bool = ...,
        legacy_serialization: bool = ...,
        **kwargs,
    ):  # -> list[str]:
        ...
    @classmethod
    def get_processor_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...
    @classmethod
    def from_args_and_dict(
        cls, args, processor_dict: dict[str, Any], **kwargs
    ):  # -> tuple[Self, dict[Any, Any]] | Self:
        ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        cache_dir: str | os.PathLike | None = ...,
        force_download: bool = ...,
        local_files_only: bool = ...,
        token: str | bool | None = ...,
        revision: str = ...,
        **kwargs,
    ) -> typing.Self: ...
    @classmethod
    def register_for_auto_class(cls, auto_class=...):  # -> None:
        ...
    @staticmethod
    def get_possibly_dynamic_module(module_name):  # -> Any:
        ...
    def batch_decode(self, *args, **kwargs): ...
    def decode(self, *args, **kwargs): ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...
    @staticmethod
    def validate_init_kwargs(
        processor_config, valid_kwargs
    ):  # -> tuple[dict[Any, Any], dict[Any, Any]]:
        ...
    @deprecate_kwarg("video_fps", version="4.58", new_name="fps")
    @deprecate_kwarg(
        "video_load_backend",
        version="4.59",
        additional_message=...,
    )
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        chat_template: str | None = ...,
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ) -> str: ...
    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=..., **kwargs
    ): ...
