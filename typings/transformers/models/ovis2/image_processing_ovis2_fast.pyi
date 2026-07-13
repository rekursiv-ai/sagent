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

class Ovis2ImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    crop_to_patches: bool | None
    min_patches: int | None
    max_patches: int | None
    use_covering_area_grid: bool | None

@auto_docstring
class Ovis2ImageProcessorFast(BaseImageProcessorFast):
    resample = ...
    image_mean = ...
    image_std = ...
    size = ...
    default_to_square = ...
    do_resize = ...
    do_rescale = ...
    do_normalize = ...
    do_convert_rgb = ...
    crop_to_patches = ...
    min_patches = ...
    max_patches = ...
    use_covering_area_grid = ...
    valid_kwargs = Ovis2ImageProcessorKwargs
    @auto_docstring
    def preprocess(
        self, images: ImageInput, **kwargs: Unpack[Ovis2ImageProcessorKwargs]
    ) -> BatchFeature: ...
    def crop_image_to_patches(
        self,
        images: torch.Tensor,
        min_patches: int,
        max_patches: int,
        use_covering_area_grid: bool = ...,
        covering_threshold: float = ...,
        patch_size: tuple | int | dict | None = ...,
        interpolation: F.InterpolationMode | None = ...,
    ):  # -> tuple[Tensor, list[list[int]]]:
        ...

__all__ = ["Ovis2ImageProcessorFast"]
