from typing import Any, overload

from PIL import Image

from .base import ChunkPipeline, build_pipeline_init_args
from ..utils import (
    ExplicitEnum,
    add_end_docstrings,
    is_pytesseract_available,
)

TESSERACT_LOADED = ...
if is_pytesseract_available():
    TESSERACT_LOADED = ...
logger = ...

def normalize_box(box, width, height):  # -> list[int]:
    ...
def apply_tesseract(
    image: Image.Image, lang: str | None, tesseract_config: str | None
):  # -> tuple[list[Any], list[Any]]:
    ...

class ModelType(ExplicitEnum):
    LayoutLM = ...
    LayoutLMv2andv3 = ...
    VisionEncoderDecoder = ...

@add_end_docstrings(
    build_pipeline_init_args(has_image_processor=True, has_tokenizer=True)
)
class DocumentQuestionAnsweringPipeline(ChunkPipeline):
    _pipeline_calls_generate = ...
    _load_processor = ...
    _load_image_processor = ...
    _load_feature_extractor = ...
    _load_tokenizer = ...
    _default_generation_config = ...
    def __init__(self, *args, **kwargs) -> None: ...
    @overload
    def __call__(
        self,
        image: Image.Image | str,
        question: str,
        word_boxes: tuple[str, list[float]] | None = ...,
        **kwargs: Any,
    ) -> list[dict[str, Any]]: ...
    @overload
    def __call__(
        self, image: dict[str, Any], **kwargs: Any
    ) -> list[dict[str, Any]]: ...
    @overload
    def __call__(
        self, image: list[dict[str, Any]], **kwargs: Any
    ) -> list[list[dict[str, Any]]]: ...
    def __call__(
        self,
        image: Image.Image | str | list[dict[str, Any]],
        question: str | None = ...,
        word_boxes: tuple[str, list[float]] | None = ...,
        **kwargs: Any,
    ) -> dict[str, Any] | list[dict[str, Any]]: ...
    def preprocess(
        self,
        input,
        padding=...,
        doc_stride=...,
        max_seq_len=...,
        word_boxes: tuple[str, list[float]] | None = ...,
        lang=...,
        tesseract_config=...,
        timeout=...,
    ):  # -> Generator[dict[str, Any | bool | None] | dict[Any | str, Tensor | list[bool | Any] | list[int | None] | Any | list[Any] | bool | None], Any, None]:
        ...
    def postprocess(self, model_outputs, top_k=..., **kwargs): ...
    def postprocess_encoder_decoder_single(
        self, model_outputs, **kwargs
    ):  # -> dict[str, None]:
        ...
    def postprocess_extractive_qa(
        self,
        model_outputs,
        top_k=...,
        handle_impossible_answer=...,
        max_answer_len=...,
        **kwargs,
    ):  # -> list[Any]:
        ...
