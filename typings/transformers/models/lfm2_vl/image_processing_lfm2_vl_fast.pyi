from functools import lru_cache

from torchvision.transforms.v2 import functional as F

import torch

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...processing_utils import Unpack
from ...utils import auto_docstring

logger = ...

def round_by_factor(number: float, factor: int) -> int: ...
def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]: ...
@lru_cache(maxsize=256)
def get_image_size_for_max_num_patches(
    image_height: int,
    image_width: int,
    patch_size: int,
    max_num_patches: int,
    eps: float = ...,
) -> tuple[int, int]: ...
def convert_image_to_patches(images: torch.Tensor, patch_size: int) -> torch.Tensor: ...
def pad_along_first_dim(
    images: torch.Tensor, target_length: int, pad_value: int = ...
) -> tuple[torch.Tensor, torch.Tensor]: ...

class Lfm2VlFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    downsample_factor: int | None
    do_image_splitting: bool | None
    min_tiles: int | None
    max_tiles: int | None
    use_thumbnail: bool | None
    min_image_tokens: int | None
    max_image_tokens: int | None
    encoder_patch_size: int | None
    tile_size: int | None
    max_pixels_tolerance: float | None
    do_pad: bool | None
    return_row_col_info: bool | None

@auto_docstring
class Lfm2VlImageProcessorFast(BaseImageProcessorFast):
    downsample_factor = ...
    do_image_splitting = ...
    min_tiles = ...
    max_tiles = ...
    use_thumbnail = ...
    min_image_tokens = ...
    max_image_tokens = ...
    encoder_patch_size = ...
    tile_size = ...
    max_pixels_tolerance = ...
    do_resize = ...
    size = ...
    resample = ...
    do_rescale = ...
    rescale_factor = ...
    do_normalize = ...
    do_pad = ...
    return_row_col_info = ...
    image_mean = ...
    image_std = ...
    valid_kwargs = Lfm2VlFastImageProcessorKwargs
    model_input_names = ...
    def __init__(self, **kwargs: Unpack[Lfm2VlFastImageProcessorKwargs]) -> None: ...
    def crop_image_to_patches(
        self,
        image: torch.Tensor,
        min_tiles: int,
        max_tiles: int,
        tile_size: int,
        use_thumbnail: bool,
        thumbnail_size: tuple[int],
        interpolation: F.InterpolationMode = ...,
        antialias: bool = ...,
        **kwargs,
    ) -> torch.Tensor: ...
    def smart_resize(
        self,
        height: int,
        width: int,
        downsample_factor: int,
        min_image_tokens: int,
        max_image_tokens: int,
        encoder_patch_size: int,
    ) -> tuple[int, int]: ...
    def resize_and_split(
        self,
        images: torch.Tensor,
        downsample_factor: int,
        min_tiles: int,
        max_tiles: int,
        use_thumbnail: bool,
        min_image_tokens: int,
        max_image_tokens: int,
        encoder_patch_size: int,
        tile_size: int,
        max_pixels_tolerance: float,
        interpolation: F.InterpolationMode,
    ) -> torch.Tensor: ...

__all__ = ["Lfm2VlImageProcessorFast"]
