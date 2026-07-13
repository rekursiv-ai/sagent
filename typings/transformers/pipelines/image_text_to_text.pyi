from typing import Any, overload

import enum

from PIL import Image

from .base import Pipeline, build_pipeline_init_args
from ..utils import add_end_docstrings

logger = ...
IMAGE_TOKEN = ...

class ReturnType(enum.Enum):
    TENSORS = ...
    NEW_TEXT = ...
    FULL_TEXT = ...

class Chat:
    def __init__(
        self,
        messages: dict,
        images: str | list[str] | Image.Image | list[Image.Image] | None = ...,
    ) -> None: ...

def add_images_to_messages(
    messages: dict,
    images: str | list[str] | Image.Image | list[Image.Image] | None,
):  # -> dict[Any, Any]:
    ...

@add_end_docstrings(build_pipeline_init_args(has_processor=True))
class ImageTextToTextPipeline(Pipeline):
    _load_processor = ...
    _load_image_processor = ...
    _load_feature_extractor = ...
    _load_tokenizer = ...
    _pipeline_calls_generate = ...
    _default_generation_config = ...
    def __init__(self, *args, **kwargs) -> None: ...
    @overload
    def __call__(
        self,
        image: str | Image.Image | None = ...,
        text: str | None = ...,
        **kwargs: Any,
    ) -> list[dict[str, Any]]: ...
    @overload
    def __call__(
        self,
        image: list[str] | list[Image.Image] | None = ...,
        text: list[str] | None = ...,
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]: ...
    def __call__(
        self,
        images: str
        | list[str]
        | list[list[str]]
        | Image.Image
        | list[Image.Image]
        | list[list[Image.Image]]
        | list[dict]
        | None = ...,
        text: str | list[str] | list[dict] | None = ...,
        **kwargs,
    ) -> list[dict[str, Any]] | list[list[dict[str, Any]]]: ...
    def preprocess(
        self, inputs=..., timeout=..., continue_final_message=..., **processing_kwargs
    ): ...
    def postprocess(
        self,
        model_outputs,
        return_type=...,
        continue_final_message=...,
        skip_special_tokens=...,
        **postprocess_kwargs,
    ):  # -> list[dict[str, str | Chat | Any]] | list[dict[str, dict[Any, Any] | str | Any]]:
        ...
