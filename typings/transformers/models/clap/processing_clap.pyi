from ...audio_utils import AudioInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils.deprecation import deprecate_kwarg

"""
Audio/Text processor class for CLAP
"""
logger = ...

class ClapProcessor(ProcessorMixin):
    feature_extractor_class = ...
    tokenizer_class = ...
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

__all__ = ["ClapProcessor"]
