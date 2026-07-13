from typing import Any, overload

from PIL import Image

from .base import Pipeline, build_pipeline_init_args
from ..utils import (
    add_end_docstrings,
)

logger = ...

@add_end_docstrings(build_pipeline_init_args(has_image_processor=True))
class ZeroShotImageClassificationPipeline(Pipeline):
    _load_processor = ...
    _load_image_processor = ...
    _load_feature_extractor = ...
    _load_tokenizer = ...
    def __init__(self, **kwargs) -> None: ...
    @overload
    def __call__(
        self, image: str | Image.Image, candidate_labels: list[str], **kwargs: Any
    ) -> list[dict[str, Any]]: ...
    @overload
    def __call__(
        self,
        image: list[str] | list[Image.Image],
        candidate_labels: list[str],
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]: ...
    def __call__(
        self,
        image: str | list[str] | Image.Image | list[Image.Image],
        candidate_labels: list[str],
        **kwargs: Any,
    ) -> list[dict[str, Any]] | list[list[dict[str, Any]]]: ...
    def preprocess(
        self,
        image,
        candidate_labels=...,
        hypothesis_template=...,
        timeout=...,
        tokenizer_kwargs=...,
    ):  # -> transformers.feature_extraction_utils.BatchFeature | Any | transformers.image_processing_base.BatchFeature:
        ...
    def postprocess(self, model_outputs):  # -> list[dict[str, Any]]:
        ...
