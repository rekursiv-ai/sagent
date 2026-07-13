from functools import lru_cache

from PIL import Image

import numpy as np

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType, filter_out_non_signature_kwargs

"""Image processor class for SigLIP2."""
logger = ...

@lru_cache(maxsize=256)
def get_image_size_for_max_num_patches(
    image_height: int,
    image_width: int,
    patch_size: int,
    max_num_patches: int,
    eps: float = ...,
) -> tuple[int, int]: ...
def convert_image_to_patches(image: np.ndarray, patch_size: int) -> np.ndarray: ...
def pad_along_first_dim(
    array: np.ndarray, target_length: int, pad_value: int = ...
) -> tuple[np.ndarray, np.ndarray]: ...

class Siglip2ImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_resize: bool = ...,
        resample: PILImageResampling = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_convert_rgb: bool | None = ...,
        patch_size: int = ...,
        max_num_patches: int = ...,
        **kwargs,
    ) -> None: ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool | None = ...,
        resample: PILImageResampling | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        return_tensors: str | TensorType | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        do_convert_rgb: bool | None = ...,
        patch_size: int | None = ...,
        max_num_patches: int | None = ...,
    ) -> Image.Image: ...

__all__ = ["Siglip2ImageProcessor"]
