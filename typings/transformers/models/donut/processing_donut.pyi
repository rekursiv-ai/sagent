from contextlib import contextmanager

from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

"""
Processor class for Donut.
"""

class DonutProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

logger = ...

class DonutProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor=..., tokenizer=..., **kwargs) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: str | list[str] | TextInput | PreTokenizedInput | None = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[DonutProcessorKwargs],
    ): ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...
    @contextmanager
    def as_target_processor(self):  # -> Generator[None, Any, None]:
        ...
    def token2json(
        self, tokens, is_inner_value=..., added_vocab=...
    ):  # -> list[dict[Any, Any]] | dict[Any, Any] | dict[str, Any] | list[Any]:
        ...
    @property
    def feature_extractor_class(self):  # -> str:
        ...
    @property
    def feature_extractor(self): ...

__all__ = ["DonutProcessor"]
