from typing import Any

from PIL import Image

from .base import GenericTensor, Pipeline, build_pipeline_init_args
from ..utils import add_end_docstrings

@add_end_docstrings(
    build_pipeline_init_args(has_image_processor=True),
    ...,
)
class ImageFeatureExtractionPipeline(Pipeline):
    _load_processor = ...
    _load_image_processor = ...
    _load_feature_extractor = ...
    _load_tokenizer = ...
    def preprocess(
        self, image, timeout=..., **image_processor_kwargs
    ) -> dict[str, GenericTensor]: ...
    def postprocess(self, model_outputs, pool=..., return_tensors=...):  # -> None:
        ...
    def __call__(
        self,
        *args: str | Image.Image | list[Image.Image] | list[str],
        **kwargs: Any,
    ) -> list[Any]: ...
