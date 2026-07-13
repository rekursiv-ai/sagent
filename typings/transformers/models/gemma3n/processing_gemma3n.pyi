import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import (
    AudioKwargs,
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput

class Gemma3nImagesKwargs(ImagesKwargs):
    do_convert_rgb: bool | None

class Gemma3nProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: AudioKwargs
    images_kwargs: Gemma3nImagesKwargs
    _defaults = ...

class Gemma3nProcessor(ProcessorMixin):
    attributes = ...
    feature_extractor_class = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        feature_extractor,
        image_processor,
        tokenizer,
        chat_template=...,
        audio_seq_length: int = ...,
        image_seq_length: int = ...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        audio: np.ndarray
        | list[float]
        | list[np.ndarray]
        | list[list[float]]
        | None = ...,
        videos=...,
        **kwargs: Unpack[Gemma3nProcessorKwargs],
    ) -> BatchFeature: ...

__all__ = ["Gemma3nProcessor"]
