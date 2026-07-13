import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

"""
Processor class for Qwen2Audio.
"""

class Qwen2AudioProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class Qwen2AudioProcessor(ProcessorMixin):
    attributes = ...
    feature_extractor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        feature_extractor=...,
        tokenizer=...,
        chat_template=...,
        audio_token=...,
        audio_bos_token=...,
        audio_eos_token=...,
    ) -> None: ...
    def __call__(
        self,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        audio: np.ndarray | list[np.ndarray] = ...,
        **kwargs: Unpack[Qwen2AudioProcessorKwargs],
    ) -> BatchFeature: ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...
    @property
    def default_chat_template(self):  # -> LiteralString:
        ...

__all__ = ["Qwen2AudioProcessor"]
