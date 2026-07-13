from collections.abc import Callable

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils.deprecation import deprecate_kwarg

"""
Processor class for IDEFICS.
"""
IMAGE_TOKEN = ...

class IdeficsImagesKwargs(ImagesKwargs, total=False):
    transform: Callable | None
    image_size: dict[str, int] | None
    image_mean: float | list[float] | None
    image_std: float | list[float] | None

class IdeficsTextKwargs(TextKwargs, total=False):
    add_eos_token: bool | None
    add_end_of_utterance_token: bool | None

class IdeficsProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: IdeficsTextKwargs
    images_kwargs: IdeficsImagesKwargs
    _defaults = ...

def incremental_to_binary_attention_mask(
    incremental_mask, return_tensors, num_classes=...
):  # -> Tensor:
    ...
def image_attention_mask_for_packed_input_ids(
    input_ids, tokenizer, return_tensors
):  # -> tuple[Tensor, Tensor] | tuple[Any, Any] | None:
    ...
def image_attention_mask_for_packed_input_ids_pt(
    input_ids, tokenizer
):  # -> tuple[Tensor, Tensor]:
    ...
def image_attention_mask_for_packed_input_ids_tf(
    input_ids, tokenizer
):  # -> tuple[Any, Any]:
    ...
def is_url(string):  # -> bool:
    ...

class IdeficsProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor,
        tokenizer=...,
        image_size=...,
        add_end_of_utterance_token=...,
        **kwargs,
    ) -> None: ...
    @deprecate_kwarg(
        old_name="prompts", version="5.0.0", new_name="text", raise_if_both_names=True
    )
    def __call__(
        self,
        images: ImageInput | list[ImageInput] | str | list[str] | list[list[str]] = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | list[list[TextInput]]
        | list[list[PreTokenizedInput]] = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[IdeficsProcessorKwargs],
    ) -> BatchFeature: ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...

__all__ = ["IdeficsProcessor"]
