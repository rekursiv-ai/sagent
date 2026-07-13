from contextlib import contextmanager

from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput

"""
Speech processor class for Wav2Vec2
"""

class Wav2Vec2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class Wav2Vec2Processor(ProcessorMixin):
    feature_extractor_class = ...
    tokenizer_class = ...
    def __init__(self, feature_extractor, tokenizer) -> None: ...
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):  # -> Self:
        ...
    def __call__(
        self,
        audio: AudioInput | None = ...,
        text: str | list[str] | TextInput | PreTokenizedInput | None = ...,
        images=...,
        videos=...,
        **kwargs: Unpack[Wav2Vec2ProcessorKwargs],
    ): ...
    def pad(self, *args, **kwargs): ...
    @property
    def model_input_names(self): ...
    @contextmanager
    def as_target_processor(self):  # -> Generator[None, Any, None]:
        ...

__all__ = ["Wav2Vec2Processor"]
