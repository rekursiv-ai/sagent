from ..auto import AutoTokenizer
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils import PreTokenizedInput, TextInput

class AriaProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class AriaProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor=...,
        tokenizer: AutoTokenizer | str = ...,
        chat_template: str | None = ...,
        size_conversion: dict[float | int, int] | None = ...,
    ) -> None: ...
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        images: ImageInput | None = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[AriaProcessorKwargs],
    ) -> BatchFeature: ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...

__all__ = ["AriaProcessor"]
