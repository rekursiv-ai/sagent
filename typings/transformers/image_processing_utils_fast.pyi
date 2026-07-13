from collections.abc import Iterable
from functools import lru_cache
from typing import Any, TypedDict

from torchvision.transforms.v2 import functional as F

import numpy as np
import torch

from .image_processing_utils import BaseImageProcessor, BatchFeature
from .image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    pil_torch_interpolation_mapping,
)
from .processing_utils import Unpack
from .utils import (
    TensorType,
    auto_docstring,
    is_torchvision_available,
)

if is_torchvision_available(): ...
else:
    pil_torch_interpolation_mapping = ...
logger = ...

@lru_cache(maxsize=10)
def validate_fast_preprocess_arguments(
    do_rescale: bool | None = ...,
    rescale_factor: float | None = ...,
    do_normalize: bool | None = ...,
    image_mean: float | list[float] | None = ...,
    image_std: float | list[float] | None = ...,
    do_center_crop: bool | None = ...,
    crop_size: SizeDict | None = ...,
    do_resize: bool | None = ...,
    size: SizeDict | None = ...,
    interpolation: F.InterpolationMode | None = ...,
    return_tensors: str | TensorType | None = ...,
    data_format: ChannelDimension = ...,
):  # -> None:
    ...
def safe_squeeze(tensor: torch.Tensor, axis: int | None = ...) -> torch.Tensor: ...
def max_across_indices(values: Iterable[Any]) -> list[Any]: ...
def get_max_height_width(images: list[torch.Tensor]) -> tuple[int, ...]: ...
def divide_to_patches(
    image: np.ndarray | torch.Tensor, patch_size: int
) -> list[np.ndarray | torch.Tensor]: ...

class DefaultFastImageProcessorKwargs(TypedDict, total=False):
    do_resize: bool | None
    size: dict[str, int] | None
    default_to_square: bool | None
    resample: PILImageResampling | F.InterpolationMode | None
    do_center_crop: bool | None
    crop_size: dict[str, int] | None
    do_rescale: bool | None
    rescale_factor: int | float | None
    do_normalize: bool | None
    image_mean: float | list[float] | None
    image_std: float | list[float] | None
    do_pad: bool | None
    pad_size: dict[str, int] | None
    do_convert_rgb: bool | None
    return_tensors: str | TensorType | None
    data_format: ChannelDimension | None
    input_data_format: str | ChannelDimension | None
    device: torch.device | None
    disable_grouping: bool | None

@auto_docstring
class BaseImageProcessorFast(BaseImageProcessor):
    resample = ...
    image_mean = ...
    image_std = ...
    size = ...
    default_to_square = ...
    crop_size = ...
    do_resize = ...
    do_center_crop = ...
    do_pad = ...
    pad_size = ...
    do_rescale = ...
    rescale_factor = ...
    do_normalize = ...
    do_convert_rgb = ...
    return_tensors = ...
    data_format = ...
    input_data_format = ...
    device = ...
    model_input_names = ...
    valid_kwargs = DefaultFastImageProcessorKwargs
    unused_kwargs = ...
    def __init__(self, **kwargs: Unpack[DefaultFastImageProcessorKwargs]) -> None: ...
    @property
    def is_fast(self) -> bool: ...
    def pad(
        self,
        images: torch.Tensor,
        pad_size: SizeDict = ...,
        fill_value: int | None = ...,
        padding_mode: str | None = ...,
        return_mask: bool = ...,
        disable_grouping: bool | None = ...,
        **kwargs,
    ) -> torch.Tensor: ...
    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        interpolation: F.InterpolationMode | None = ...,
        antialias: bool = ...,
        **kwargs,
    ) -> torch.Tensor: ...
    @staticmethod
    def compile_friendly_resize(
        image: torch.Tensor,
        new_size: tuple[int, int],
        interpolation: F.InterpolationMode | None = ...,
        antialias: bool = ...,
    ) -> torch.Tensor: ...
    def rescale(self, image: torch.Tensor, scale: float, **kwargs) -> torch.Tensor: ...
    def normalize(
        self,
        image: torch.Tensor,
        mean: float | Iterable[float],
        std: float | Iterable[float],
        **kwargs,
    ) -> torch.Tensor: ...
    def rescale_and_normalize(
        self,
        images: torch.Tensor,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float],
        image_std: float | list[float],
    ) -> torch.Tensor: ...
    def center_crop(
        self, image: torch.Tensor, size: SizeDict, **kwargs
    ) -> torch.Tensor: ...
    def convert_to_rgb(self, image: ImageInput) -> ImageInput: ...
    def filter_out_unused_kwargs(self, kwargs: dict):  # -> dict[Any, Any]:
        ...
    def __call__(
        self,
        images: ImageInput,
        *args,
        **kwargs: Unpack[DefaultFastImageProcessorKwargs],
    ) -> BatchFeature: ...
    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        *args,
        **kwargs: Unpack[DefaultFastImageProcessorKwargs],
    ) -> BatchFeature: ...
    def to_dict(self):  # -> dict[str, Any]:
        ...
