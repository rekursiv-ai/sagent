from functools import lru_cache

from torchvision.transforms.v2 import functional as F

import torch

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...image_utils import ImageInput
from ...processing_utils import Unpack
from ...utils import auto_docstring

class Cohere2VisionFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    crop_to_patches: bool | None
    min_patches: int | None
    max_patches: int | None

@lru_cache(maxsize=10)
def get_all_supported_aspect_ratios(max_image_tiles: int) -> list[tuple[int, int]]: ...
def get_optimal_tiled_canvas(
    original_image_size: tuple[int, int],
    target_tile_size: tuple[int, int],
    min_image_tiles: int,
    max_image_tiles: int,
) -> tuple[int, int]: ...

@auto_docstring
class Cohere2VisionImageProcessorFast(BaseImageProcessorFast):
    resample = ...
    image_mean = ...
    image_std = ...
    size = ...
    do_resize = ...
    do_rescale = ...
    do_normalize = ...
    do_convert_rgb = ...
    crop_to_patches = ...
    min_patches = ...
    max_patches = ...
    valid_kwargs = Cohere2VisionFastImageProcessorKwargs
    patch_size = ...
    def __init__(
        self, **kwargs: Unpack[Cohere2VisionFastImageProcessorKwargs]
    ) -> None: ...
    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[Cohere2VisionFastImageProcessorKwargs],
    ) -> BatchFeature: ...
    def crop_image_to_patches(
        self,
        images: torch.Tensor,
        min_patches: int,
        max_patches: int,
        use_thumbnail: bool = ...,
        patch_size: tuple | int | dict | None = ...,
        interpolation: F.InterpolationMode | None = ...,
    ):  # -> Tensor:
        ...
    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs=...
    ):  # -> int:
        ...

__all__ = ["Cohere2VisionImageProcessorFast"]
