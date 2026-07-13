from ...audio_utils import AudioInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

logger = ...

class ParakeetProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class ParakeetProcessor(ProcessorMixin):
    attributes = ...
    feature_extractor_class = ...
    tokenizer_class = ...
    def __call__(
        self,
        audio: AudioInput,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = ...,
        sampling_rate: int | None = ...,
        **kwargs: Unpack[ParakeetProcessorKwargs],
    ): ...
    @property
    def model_input_names(self): ...

__all__ = ["ParakeetProcessor"]
