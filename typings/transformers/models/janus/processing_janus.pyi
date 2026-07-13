from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

"""
Processor class for Janus.
"""
logger = ...
DEFAULT_SYSTEM_PROMPT = ...

class JanusTextKwargs(TextKwargs, total=False):
    generation_mode: str

class JanusProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: JanusTextKwargs
    _defaults = ...

class JanusProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template=...,
        use_default_system_prompt=...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        images: ImageInput | None = ...,
        videos=...,
        audio=...,
        **kwargs: Unpack[JanusProcessorKwargs],
    ) -> BatchFeature: ...
    def postprocess(self, images: ImageInput, **kwargs): ...

__all__ = ["JanusProcessor"]
