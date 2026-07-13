import torch

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

class ColPaliProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

IMAGE_TOKEN = ...
EXTRA_TOKENS = ...

def build_string_from_input(
    prompt, bos_token, image_seq_len, image_token, num_images
):  # -> str:
    ...

class ColPaliProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor=...,
        tokenizer=...,
        chat_template=...,
        visual_prompt_prefix: str = ...,
        query_prefix: str = ...,
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
        **kwargs: Unpack[ColPaliProcessorKwargs],
    ) -> BatchFeature: ...
    @property
    def query_augmentation_token(self) -> str: ...
    def process_images(
        self,
        images: ImageInput | None = ...,
        **kwargs: Unpack[ColPaliProcessorKwargs],
    ) -> BatchFeature: ...
    def process_queries(
        self,
        text: TextInput | list[TextInput],
        **kwargs: Unpack[ColPaliProcessorKwargs],
    ) -> BatchFeature: ...
    def score_retrieval(
        self,
        query_embeddings: torch.Tensor | list[torch.Tensor],
        passage_embeddings: torch.Tensor | list[torch.Tensor],
        batch_size: int = ...,
        output_dtype: torch.dtype | None = ...,
        output_device: torch.device | str = ...,
    ) -> torch.Tensor: ...

__all__ = ["ColPaliProcessor"]
