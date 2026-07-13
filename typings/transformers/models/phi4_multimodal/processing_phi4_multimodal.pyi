from ...audio_utils import AudioInput
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput

"""
Processor class for Phi4Multimodal
"""
logger = ...

class Phi4MultimodalProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class Phi4MultimodalProcessor(ProcessorMixin):
    attributes = ...
    tokenizer_class = ...
    image_processor_class = ...
    audio_processor_class = ...
    def __init__(
        self, image_processor, audio_processor, tokenizer, **kwargs
    ) -> None: ...
    def __call__(
        self,
        text: TextInput | list[TextInput],
        images: ImageInput | None = ...,
        audio: AudioInput | None = ...,
        **kwargs: Unpack[ProcessingKwargs],
    ) -> BatchFeature: ...

__all__ = ["Phi4MultimodalProcessor"]
