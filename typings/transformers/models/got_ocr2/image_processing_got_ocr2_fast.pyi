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

"""Fast Image processor class for Got-OCR-2."""

class GotOcr2FastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    crop_to_patches: bool | None
    min_patches: int | None
    max_patches: int | None

@auto_docstring
class GotOcr2ImageProcessorFast(BaseImageProcessorFast):
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
    valid_kwargs = GotOcr2FastImageProcessorKwargs
    def __init__(self, **kwargs: Unpack[GotOcr2FastImageProcessorKwargs]) -> None: ...
    @auto_docstring
    def preprocess(
        self, images: ImageInput, **kwargs: Unpack[GotOcr2FastImageProcessorKwargs]
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

__all__ = ["GotOcr2ImageProcessorFast"]
