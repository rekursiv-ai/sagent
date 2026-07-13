from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
)
from ...tokenization_utils_base import TextInput

"""Processor class for KOSMOS-2."""
type BboxInput = (
    list[tuple[int, int]]
    | list[tuple[float, float, float, float]]
    | list[list[tuple[int, int]]]
    | list[list[tuple[float, float, float]]]
)

class Kosmos2ImagesKwargs(ImagesKwargs, total=False):
    bboxes: list[float] | None
    num_image_tokens: int | None
    first_image_token_id: int | None

class Kosmos2TextKwargs(TextKwargs, total=False):
    add_eos_token: bool | None

class Kosmos2ProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: Kosmos2TextKwargs
    images_kwargs: Kosmos2ImagesKwargs
    _defaults = ...

class Kosmos2Processor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self, image_processor, tokenizer, num_patch_index_tokens=..., *kwargs
    ) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput | list[TextInput] = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[Kosmos2ProcessorKwargs],
    ) -> BatchFeature: ...
    def preprocess_examples(
        self,
        texts: TextInput | list[TextInput],
        images: ImageInput | None = ...,
        bboxes: BboxInput = ...,
        num_image_tokens: int | None = ...,
    ) -> str | list[str]: ...
    def post_process_generation(
        self, text, cleanup_and_extract=...
    ):  # -> tuple[str, list[Any]]:
        ...
    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=..., **kwargs
    ):  # -> list[tuple[str, list[Any]] | Any]:
        ...
    @property
    def model_input_names(self): ...

def coordinate_to_patch_index(
    bbox: tuple[float, float, float, float], num_patches_per_side: int
) -> tuple[int, int]: ...
def patch_index_to_coordinate(
    ul_idx: int, lr_idx: int, num_patches_per_side: int
):  # -> tuple[float, float, float, float]:
    ...
def extract_entities_with_patch_indices(text):  # -> list[Any]:
    ...
def adjust_entity_positions(entity, text):  # -> tuple[Any, tuple[int, int]]:
    ...
def clean_text_and_extract_entities_with_bboxes(
    text, num_patches_per_side=...
):  # -> tuple[str, list[Any]]:
    ...

__all__ = ["Kosmos2Processor"]
