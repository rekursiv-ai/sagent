from ...audio_utils import AudioInput
from ...processing_utils import (
    AllKwargsForChatTemplate,
    AudioKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput

logger = ...

class VoxtralAudioKwargs(AudioKwargs, total=False):
    max_source_positions: int | None

class VoxtralProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class VoxtralProcessor(ProcessorMixin):
    attributes = ...
    feature_extractor_class = ...
    tokenizer_class = ...
    def __init__(self, feature_extractor, tokenizer) -> None: ...
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ) -> str: ...
    def __call__(
        self,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None,
        **kwargs: Unpack[VoxtralProcessorKwargs],
    ):  # -> BatchFeature:
        ...
    def apply_transcription_request(
        self,
        language: str | list[str],
        audio: str | list[str] | AudioInput,
        model_id: str,
        sampling_rate: int | None = ...,
        format: str | list[str] | None = ...,
        **kwargs: Unpack[VoxtralProcessorKwargs],
    ):  # -> BatchFeature | list[Any]:
        ...

__all__ = ["VoxtralProcessor"]
