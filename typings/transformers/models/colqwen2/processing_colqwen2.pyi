import torch

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

class ColQwen2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class ColQwen2Processor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor=...,
        tokenizer=...,
        chat_template=...,
        visual_prompt_prefix: str | None = ...,
        query_prefix: str | None = ...,
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
        **kwargs: Unpack[ColQwen2ProcessorKwargs],
    ) -> BatchFeature: ...
    @property
    def query_augmentation_token(self) -> str: ...
    def process_images(
        self,
        images: ImageInput | None = ...,
        **kwargs: Unpack[ColQwen2ProcessorKwargs],
    ) -> BatchFeature: ...
    def process_queries(
        self,
        text: TextInput | list[TextInput],
        **kwargs: Unpack[ColQwen2ProcessorKwargs],
    ) -> BatchFeature: ...
    def score_retrieval(
        self,
        query_embeddings: torch.Tensor | list[torch.Tensor],
        passage_embeddings: torch.Tensor | list[torch.Tensor],
        batch_size: int = ...,
        output_dtype: torch.dtype | None = ...,
        output_device: torch.device | str = ...,
    ) -> torch.Tensor: ...
    @property
    def model_input_names(self): ...

__all__ = ["ColQwen2Processor"]
