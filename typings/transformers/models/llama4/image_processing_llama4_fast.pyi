from functools import lru_cache

import torch

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...image_utils import ImageInput, SizeDict
from ...processing_utils import Unpack
from ...utils import auto_docstring

"""Fast Image processor class for Got-OCR-2."""

def get_factors(dividend: int) -> set[int]: ...
def get_max_res_without_distortion(
    image_size: tuple[int, int], target_size: tuple[int, int]
) -> tuple[int, int]: ...
def split_to_tiles(
    images: torch.Tensor, num_tiles_height: int, num_tiles_width: int
) -> torch.Tensor: ...
@lru_cache(maxsize=1)
def find_supported_resolutions(
    max_num_chunks: int, patch_size: SizeDict
) -> torch.Tensor: ...
def pad_to_best_fit(
    images: torch.Tensor,
    target_size: tuple[int, int],
    background_color: int | tuple[int, int, int] = ...,
) -> torch.Tensor: ...
def get_best_fit(
    image_size: tuple[int, int],
    possible_resolutions: torch.Tensor,
    resize_to_max_canvas: bool = ...,
) -> tuple[int, int]: ...

class Llama4ImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    max_patches: int | None
    resize_to_max_canvas: bool | None

@auto_docstring
class Llama4ImageProcessorFast(BaseImageProcessorFast):
    resample = ...
    image_mean = ...
    image_std = ...
    size = ...
    do_resize = ...
    do_rescale = ...
    do_normalize = ...
    do_convert_rgb = ...
    max_patches = ...
    resize_to_max_canvas = ...
    valid_kwargs = Llama4ImageProcessorKwargs
    def __init__(self, **kwargs: Unpack[Llama4ImageProcessorKwargs]) -> None: ...
    @torch.compiler.disable
    def rescale_and_normalize(
        self,
        images: torch.Tensor,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float],
        image_std: float | list[float],
    ) -> torch.Tensor: ...
    @auto_docstring
    def preprocess(
        self, images: ImageInput, **kwargs: Unpack[Llama4ImageProcessorKwargs]
    ) -> BatchFeature: ...

__all__ = ["Llama4ImageProcessorFast"]
