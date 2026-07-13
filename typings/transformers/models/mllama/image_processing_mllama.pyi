from functools import lru_cache

import numpy as np

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
)
from ...utils import TensorType

logger = ...

@lru_cache(maxsize=10)
def get_all_supported_aspect_ratios(max_image_tiles: int) -> list[tuple[int, int]]: ...
def get_image_size_fit_to_canvas(
    image_height: int,
    image_width: int,
    canvas_height: int,
    canvas_width: int,
    tile_size: int,
) -> tuple[int, int]: ...
@lru_cache(maxsize=100)
def get_optimal_tiled_canvas(
    image_height: int, image_width: int, max_image_tiles: int, tile_size: int
) -> tuple[int, int]: ...
def split_to_tiles(
    image: np.ndarray, num_tiles_height: int, num_tiles_width: int
) -> np.ndarray: ...
def build_aspect_ratio_mask(
    aspect_ratios: list[list[tuple[int, int]]], max_image_tiles: int
) -> np.ndarray: ...
def pack_images(
    batch_images: list[list[np.ndarray]], max_image_tiles: int
) -> tuple[np.ndarray, list[list[int]]]: ...
def pack_aspect_ratios(
    aspect_ratios: list[list[tuple[int, int]]], pad_value: int = ...
) -> np.ndarray: ...
def convert_aspect_ratios_to_ids(
    aspect_ratios: list[list[tuple[int, int]]], max_image_tiles: int
) -> np.ndarray: ...
def to_channel_dimension_format(
    image: np.ndarray,
    channel_dim: ChannelDimension | str,
    input_channel_dim: ChannelDimension | str | None = ...,
) -> np.ndarray: ...
def convert_to_rgb(image: ImageInput) -> ImageInput: ...

class MllamaImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_convert_rgb: bool = ...,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_pad: bool = ...,
        max_image_tiles: int = ...,
        **kwargs,
    ) -> None: ...
    def preprocess(
        self,
        images: ImageInput,
        do_convert_rgb: bool | None = ...,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_pad: bool | None = ...,
        max_image_tiles: int | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        return_tensors: str | TensorType | None = ...,
    ):  # -> BatchFeature:
        ...
    def pad(
        self,
        image: np.ndarray,
        size: dict[str, int],
        aspect_ratio: tuple[int, int],
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> np.ndarray: ...
    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        max_image_tiles: int,
        resample: PILImageResampling = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> np.ndarray | tuple[int, int]: ...

__all__ = ["MllamaImageProcessor"]
