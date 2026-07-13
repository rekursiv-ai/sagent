import torch

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_utils import PreTokenizedInput, TextInput

"""Processor class for Granite Speech."""
logger = ...

class GraniteSpeechProcessor(ProcessorMixin):
    attributes = ...
    audio_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self, audio_processor, tokenizer, audio_token=..., chat_template=...
    ) -> None: ...
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        audio: torch.Tensor | list[torch.Tensor] = ...,
        device: str = ...,
        images=...,
        videos=...,
        **kwargs,
    ) -> BatchFeature: ...

__all__ = ["GraniteSpeechProcessor"]
