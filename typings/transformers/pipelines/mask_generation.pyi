from typing import Any, overload

from PIL import Image

from .base import ChunkPipeline, build_pipeline_init_args
from ..utils import add_end_docstrings

logger = ...

@add_end_docstrings(
    build_pipeline_init_args(has_image_processor=True),
    ...,
)
class MaskGenerationPipeline(ChunkPipeline):
    _load_processor = ...
    _load_image_processor = ...
    _load_feature_extractor = ...
    _load_tokenizer = ...
    def __init__(self, **kwargs) -> None: ...
    @overload
    def __call__(
        self, image: str | Image.Image, *args: Any, **kwargs: Any
    ) -> dict[str, Any]: ...
    @overload
    def __call__(
        self, image: list[str] | list[Image.Image], *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]]: ...
    def __call__(
        self,
        image: str | Image.Image | list[str] | list[Image.Image],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any] | list[dict[str, Any]]: ...
    def preprocess(
        self,
        image,
        points_per_batch=...,
        crops_n_layers: int = ...,
        crop_overlap_ratio: float = ...,
        points_per_crop: int = ...,
        crop_n_points_downscale_factor: int = ...,
        timeout: float | None = ...,
    ):  # -> Generator[dict[str | Any, Any], Any, None]:
        ...
    def postprocess(
        self,
        model_outputs,
        output_rle_mask=...,
        output_bboxes_mask=...,
        crops_nms_thresh=...,
    ):  # -> dict[str | Any, Any | list[Any]]:
        ...
