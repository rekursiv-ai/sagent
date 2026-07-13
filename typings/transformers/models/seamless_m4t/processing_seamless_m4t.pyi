from ...audio_utils import AudioInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils.deprecation import deprecate_kwarg

"""
Audio/Text processor class for SeamlessM4T
"""
logger = ...

class SeamlessM4TTextKwargs(TextKwargs):
    src_lang: str | None
    tgt_lang: str | None

class SeamlessM4TProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: SeamlessM4TTextKwargs
    _defaults = ...

class SeamlessM4TProcessor(ProcessorMixin):
    feature_extractor_class = ...
    tokenizer_class = ...
    valid_processor_kwargs = SeamlessM4TProcessorKwargs
    def __init__(self, feature_extractor, tokenizer) -> None: ...
    @deprecate_kwarg("audios", version="v4.59.0", new_name="audio")
    def __call__(
        self,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = ...,
        audios: AudioInput | None = ...,
        audio: AudioInput | None = ...,
        **kwargs: Unpack[ProcessingKwargs],
    ):  # -> BatchFeature:
        ...

__all__ = ["SeamlessM4TProcessor"]
