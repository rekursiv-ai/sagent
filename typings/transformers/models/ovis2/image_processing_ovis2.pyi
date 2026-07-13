from functools import lru_cache

import numpy as np
import PIL

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType, filter_out_non_signature_kwargs

logger = ...

@lru_cache(maxsize=10)
def get_all_supported_aspect_ratios(
    min_image_tiles: int, max_image_tiles: int
) -> list[tuple[int, int]]: ...
@lru_cache(maxsize=100)
def get_optimal_tiled_canvas(
    original_image_size: tuple[int, int],
    target_tile_size: tuple[int, int],
    min_image_tiles: int,
    max_image_tiles: int,
) -> tuple[int, int]: ...
def compute_patch_covering_area(
    left: int, upper: int, right: int, lower: int, side: int
) -> float: ...
def split_image_into_grid(
    h: int, w: int, grid: tuple[int, int]
) -> list[tuple[int, int, int, int]]: ...
@lru_cache(maxsize=100)
def get_min_tile_covering_grid(
    image_size: tuple[int, int],
    target_patch_size: int,
    max_image_tiles: int,
    covering_threshold: float = ...,
) -> tuple[int, int]: ...

class Ovis2ImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        crop_to_patches: bool = ...,
        min_patches: int = ...,
        max_patches: int = ...,
        resample: PILImageResampling = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_convert_rgb: bool = ...,
        use_covering_area_grid: bool = ...,
        **kwargs,
    ) -> None: ...
    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        resample: PILImageResampling = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        crop_to_patches: bool | None = ...,
        min_patches: int | None = ...,
        max_patches: int | None = ...,
        resample: PILImageResampling | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        return_tensors: str | TensorType | None = ...,
        do_convert_rgb: bool | None = ...,
        data_format: ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
        use_covering_area_grid: bool = ...,
    ) -> PIL.Image.Image: ...
    def crop_image_to_patches(
        self,
        images: np.ndarray,
        min_patches: int,
        max_patches: int,
        use_covering_area_grid: bool = ...,
        patch_size: tuple | int | dict | None = ...,
        data_format: ChannelDimension | None = ...,
        covering_threshold: float = ...,
    ):  # -> tuple[list[Any], tuple[int, int]]:
        ...

__all__ = ["Ovis2ImageProcessor"]
