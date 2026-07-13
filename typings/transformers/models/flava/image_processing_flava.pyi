from collections.abc import Iterable
from functools import lru_cache
from typing import Any

import numpy as np
import PIL

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType, filter_out_non_signature_kwargs
from ...utils.import_utils import requires

"""Image processor class for Flava."""
logger = ...
FLAVA_IMAGE_MEAN = ...
FLAVA_IMAGE_STD = ...
FLAVA_CODEBOOK_MEAN = ...
FLAVA_CODEBOOK_STD = ...
LOGIT_LAPLACE_EPS: float = ...

class FlavaMaskingGenerator:
    def __init__(
        self,
        input_size: int | tuple[int, int] = ...,
        total_mask_patches: int = ...,
        mask_group_max_patches: int | None = ...,
        mask_group_min_patches: int = ...,
        mask_group_min_aspect_ratio: float | None = ...,
        mask_group_max_aspect_ratio: float | None = ...,
    ) -> None: ...
    def __repr__(self):  # -> str:
        ...
    def get_shape(self):  # -> tuple[Any | int, Any | int]:
        ...
    def __call__(self):  # -> NDArray[Any]:
        ...

@requires(backends=("vision",))
class FlavaImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        do_center_crop: bool = ...,
        crop_size: dict[str, int] | None = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | Iterable[float] | None = ...,
        image_std: float | Iterable[float] | None = ...,
        return_image_mask: bool = ...,
        input_size_patches: int = ...,
        total_mask_patches: int = ...,
        mask_group_min_patches: int = ...,
        mask_group_max_patches: int | None = ...,
        mask_group_min_aspect_ratio: float = ...,
        mask_group_max_aspect_ratio: float | None = ...,
        return_codebook_pixels: bool = ...,
        codebook_do_resize: bool = ...,
        codebook_size: bool | None = ...,
        codebook_resample: int = ...,
        codebook_do_center_crop: bool = ...,
        codebook_crop_size: int | None = ...,
        codebook_do_rescale: bool = ...,
        codebook_rescale_factor: float = ...,
        codebook_do_map_pixels: bool = ...,
        codebook_do_normalize: bool = ...,
        codebook_image_mean: float | Iterable[float] | None = ...,
        codebook_image_std: float | Iterable[float] | None = ...,
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_dict(
        cls, image_processor_dict: dict[str, Any], **kwargs
    ):  # -> tuple[Self, dict[str, Any]] | Self:
        ...
    @lru_cache  # noqa: B019 -- mirrors upstream @lru_cache on a method; not executed in a stub
    def masking_generator(
        self,
        input_size_patches,
        total_mask_patches,
        mask_group_min_patches,
        mask_group_max_patches,
        mask_group_min_aspect_ratio,
        mask_group_max_aspect_ratio,
    ) -> FlavaMaskingGenerator: ...
    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        resample: PILImageResampling = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    def map_pixels(self, image: np.ndarray) -> np.ndarray: ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling | None = ...,
        do_center_crop: bool | None = ...,
        crop_size: dict[str, int] | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        return_image_mask: bool | None = ...,
        input_size_patches: int | None = ...,
        total_mask_patches: int | None = ...,
        mask_group_min_patches: int | None = ...,
        mask_group_max_patches: int | None = ...,
        mask_group_min_aspect_ratio: float | None = ...,
        mask_group_max_aspect_ratio: float | None = ...,
        return_codebook_pixels: bool | None = ...,
        codebook_do_resize: bool | None = ...,
        codebook_size: dict[str, int] | None = ...,
        codebook_resample: int | None = ...,
        codebook_do_center_crop: bool | None = ...,
        codebook_crop_size: dict[str, int] | None = ...,
        codebook_do_rescale: bool | None = ...,
        codebook_rescale_factor: float | None = ...,
        codebook_do_map_pixels: bool | None = ...,
        codebook_do_normalize: bool | None = ...,
        codebook_image_mean: Iterable[float] | None = ...,
        codebook_image_std: Iterable[float] | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> PIL.Image.Image: ...

__all__ = ["FlavaImageProcessor"]
