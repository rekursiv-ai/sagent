from collections.abc import Iterable

import numpy as np

from ...image_processing_utils import BaseImageProcessor
from ...image_transforms import PaddingMode
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType

logger = ...

def divide_to_patches(
    image: np.ndarray, patch_size: int, input_data_format
) -> list[np.ndarray]: ...

class AriaImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        image_mean: list[float] | None = ...,
        image_std: list[float] | None = ...,
        max_image_size: int = ...,
        min_image_size: int = ...,
        split_resolutions: list[tuple[int, int]] | None = ...,
        split_image: bool | None = ...,
        do_convert_rgb: bool | None = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool | None = ...,
        resample: PILImageResampling = ...,
        **kwargs,
    ) -> None: ...
    def preprocess(
        self,
        images: ImageInput | list[ImageInput],
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        max_image_size: int | None = ...,
        min_image_size: int | None = ...,
        split_image: bool | None = ...,
        do_convert_rgb: bool | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        resample: PILImageResampling | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> BatchFeature:
        ...
    def pad(
        self,
        image: np.ndarray,
        padding: int | tuple[int, int] | Iterable[tuple[int, int]],
        mode: PaddingMode = ...,
        constant_values: float | Iterable[float] = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> np.ndarray: ...
    def get_image_patches(
        self,
        image: np.ndarray,
        grid_pinpoints: list[tuple[int, int]],
        patch_size: int,
        resample: PILImageResampling,
        data_format: ChannelDimension,
        input_data_format: ChannelDimension,
    ) -> list[np.ndarray]: ...
    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs=...
    ):  # -> Literal[1]:
        ...

__all__ = ["AriaImageProcessor"]
