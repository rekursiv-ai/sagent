from typing import Any

import torch

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

logger = ...

class Florence2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class Florence2Processor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor=...,
        tokenizer=...,
        num_additional_image_tokens: int = ...,
        post_processor_config: dict | None = ...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        **kwargs: Unpack[Florence2ProcessorKwargs],
    ) -> BatchFeature: ...
    def batch_decode(self, *args, **kwargs): ...
    def decode(self, *args, **kwargs): ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...
    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=..., **kwargs
    ): ...
    def post_process_generation(
        self, text=..., sequence=..., task=..., image_size=...
    ) -> dict[str, Any]: ...

class Florence2PostProcessor:
    def __init__(self, config, tokenizer) -> None: ...
    def quantize(
        self, locations: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor: ...
    def dequantize(
        self, locations: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor: ...
    def decode_with_spans(
        self, token_ids: list[int]
    ) -> tuple[str, list[tuple[int, int]]]: ...
    def parse_ocr_from_text_and_spans(
        self,
        text: str,
        pattern: str | None,
        image_size: tuple[int, int],
        area_threshold: float = ...,
    ) -> list[dict[str, Any]]: ...
    def parse_phrase_grounding_from_text_and_spans(
        self, text: str, image_size: tuple[int, int]
    ) -> list[dict[str, Any]]: ...
    def parse_description_with_bboxes_from_text_and_spans(
        self, text: str, image_size: tuple[int, int], allow_empty_phrase: bool = ...
    ) -> list[dict[str, Any]]: ...
    def parse_description_with_polygons_from_text_and_spans(
        self,
        text: str,
        image_size: tuple[int, int],
        allow_empty_phrase: bool = ...,
        polygon_sep_token: str = ...,
        polygon_start_token: str = ...,
        polygon_end_token: str = ...,
        with_box_at_start: bool = ...,
    ) -> list[dict[str, Any]]: ...
    def __call__(
        self, text=..., sequence=..., image_size=..., parse_tasks=...
    ) -> dict[str, Any]: ...

__all__ = ["Florence2Processor"]
