from typing import Any

import os

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_utils import ImageInput
from ...utils import TensorType
from ...utils.import_utils import requires

logger = ...

@requires(backends=("torch", "timm", "torchvision"))
class TimmWrapperImageProcessor(BaseImageProcessor):
    main_input_name = ...
    def __init__(
        self,
        pretrained_cfg: dict[str, Any],
        architecture: str | None = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def get_image_processor_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...
    def preprocess(
        self, images: ImageInput, return_tensors: str | TensorType | None = ...
    ) -> BatchFeature: ...
    def save_pretrained(self, *args, **kwargs):  # -> None:
        ...

__all__ = ["TimmWrapperImageProcessor"]
