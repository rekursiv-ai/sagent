from pathlib import Path
from typing import Any

from ...audio_utils import AudioInput
from ...processing_utils import AudioKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

class CsmAudioKwargs(AudioKwargs, total=False):
    encoded_length_kwargs: dict[str, Any] | None

class CsmProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: CsmAudioKwargs
    _defaults = ...

class CsmProcessor(ProcessorMixin):
    attributes = ...
    feature_extractor_class = ...
    tokenizer_class = ...
    def __init__(self, feature_extractor, tokenizer, chat_template=...) -> None: ...
    def save_audio(
        self,
        audio: AudioInput,
        saving_path: str | Path | list[str | Path],
        **kwargs: Unpack[CsmProcessorKwargs],
    ):  # -> None:
        ...
    def __call__(
        self,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None,
        audio: AudioInput | None = ...,
        output_labels: bool | None = ...,
        depth_decoder_labels_ratio: float | None = ...,
        **kwargs: Unpack[CsmProcessorKwargs],
    ):  # -> BatchFeature:
        ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...

__all__ = ["CsmProcessor"]
