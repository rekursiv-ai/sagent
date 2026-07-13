from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

"""
Processor class for InstructBLIP. Largely copy of Blip2Processor with addition of a tokenizer for the Q-Former.
"""
logger = ...

class InstructBlipProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class InstructBlipProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    qformer_tokenizer_class = ...
    def __init__(
        self,
        image_processor,
        tokenizer,
        qformer_tokenizer,
        num_query_tokens=...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[InstructBlipProcessorKwargs],
    ) -> BatchFeature: ...
    @property
    def model_input_names(self): ...
    def save_pretrained(self, save_directory, **kwargs):  # -> list[str]:
        ...
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, **kwargs
    ):  # -> InstructBlipProcessor:
        ...

__all__ = ["InstructBlipProcessor"]
