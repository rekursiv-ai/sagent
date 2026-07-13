from collections.abc import Iterable

import numpy as np

from .image_processing_base import BatchFeature, ImageProcessingMixin
from .image_utils import ChannelDimension
from .utils.import_utils import requires

logger = ...
INIT_SERVICE_KWARGS = ...

@requires(backends=("vision",))
class BaseImageProcessor(ImageProcessingMixin):
    def __init__(self, **kwargs) -> None: ...
    @property
    def is_fast(self) -> bool: ...
    def __call__(self, images, **kwargs) -> BatchFeature: ...
    def preprocess(self, images, **kwargs) -> BatchFeature: ...
    def rescale(
        self,
        image: np.ndarray,
        scale: float,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    def normalize(
        self,
        image: np.ndarray,
        mean: float | Iterable[float],
        std: float | Iterable[float],
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    def center_crop(
        self,
        image: np.ndarray,
        size: dict[str, int],
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    def to_dict(self):  # -> dict[str, Any]:
        ...

VALID_SIZE_DICT_KEYS = ...

def is_valid_size_dict(size_dict):  # -> bool:
    ...
def convert_to_size_dict(
    size,
    max_size: int | None = ...,
    default_to_square: bool = ...,
    height_width_order: bool = ...,
):  # -> dict[str, int] | dict[str, Any]:
    ...
def get_size_dict(
    size: int | Iterable[int] | dict[str, int] | None = ...,
    max_size: int | None = ...,
    height_width_order: bool = ...,
    default_to_square: bool = ...,
    param_name=...,
) -> dict: ...
def select_best_resolution(
    original_size: tuple, possible_resolutions: list
) -> tuple: ...
def get_patch_output_size(
    image, target_resolution, input_data_format
):  # -> tuple[Any, Any]:
    ...
