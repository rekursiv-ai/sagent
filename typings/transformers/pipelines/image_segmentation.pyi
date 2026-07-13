from typing import Any, overload

from PIL import Image

from .base import Pipeline, build_pipeline_init_args
from ..utils import add_end_docstrings

logger = ...

@add_end_docstrings(build_pipeline_init_args(has_image_processor=True))
class ImageSegmentationPipeline(Pipeline):
    _load_processor = ...
    _load_image_processor = ...
    _load_feature_extractor = ...
    _load_tokenizer = ...
    def __init__(self, *args, **kwargs) -> None: ...
    @overload
    def __call__(
        self, inputs: str | Image.Image, **kwargs: Any
    ) -> list[dict[str, Any]]: ...
    @overload
    def __call__(
        self, inputs: list[str] | list[Image.Image], **kwargs: Any
    ) -> list[list[dict[str, Any]]]: ...
    def __call__(
        self,
        inputs: str | Image.Image | list[str] | list[Image.Image],
        **kwargs: Any,
    ) -> list[dict[str, Any]] | list[list[dict[str, Any]]]: ...
    def preprocess(
        self, image, subtask=..., timeout=...
    ):  # -> transformers.feature_extraction_utils.BatchFeature | Any | transformers.image_processing_base.BatchFeature:
        ...
    def postprocess(
        self,
        model_outputs,
        subtask=...,
        threshold=...,
        mask_threshold=...,
        overlap_mask_area_threshold=...,
    ):  # -> list[Any]:
        ...
