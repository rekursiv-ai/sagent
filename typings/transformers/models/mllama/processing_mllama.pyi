import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

"""Processor class for Mllama."""

class MllamaImagesKwargs(ImagesKwargs, total=False):
    max_image_tiles: int | None

class MllamaProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: MllamaImagesKwargs
    _defaults = ...

def get_cross_attention_token_mask(
    input_ids: list[int], image_token_id: int
) -> list[list[int]]: ...
def convert_sparse_cross_attention_mask_to_dense(
    cross_attention_token_mask: list[list[list[int]]],
    num_tiles: list[list[int]],
    max_num_tiles: int,
    length: int,
) -> np.ndarray: ...
def build_string_from_input(prompt: str, bos_token: str, image_token: str) -> str: ...

class MllamaProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor, tokenizer, chat_template=...) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[MllamaProcessorKwargs],
    ) -> BatchFeature: ...
    def post_process_image_text_to_text(
        self,
        generated_outputs,
        skip_special_tokens=...,
        clean_up_tokenization_spaces=...,
        **kwargs,
    ): ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...

__all__ = ["MllamaProcessor"]
