from torchvision.transforms.v2 import functional as F

import torch

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    BatchFeature,
    DefaultFastImageProcessorKwargs,
)
from ...image_utils import ImageInput, SizeDict
from ...processing_utils import Unpack
from ...utils import auto_docstring

"""Fast Image processor class for PoolFormer."""

class PoolFormerFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    crop_pct: float | None

@auto_docstring
class PoolFormerImageProcessorFast(BaseImageProcessorFast):
    resample = ...
    image_mean = ...
    image_std = ...
    size = ...
    default_to_square = ...
    crop_size = ...
    crop_pct = ...
    do_resize = ...
    do_center_crop = ...
    do_rescale = ...
    do_normalize = ...
    valid_kwargs = PoolFormerFastImageProcessorKwargs
    def __init__(
        self, **kwargs: Unpack[PoolFormerFastImageProcessorKwargs]
    ) -> None: ...
    @auto_docstring
    def preprocess(
        self, images: ImageInput, **kwargs: Unpack[PoolFormerFastImageProcessorKwargs]
    ) -> BatchFeature: ...
    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        crop_pct: float | None = ...,
        interpolation: F.InterpolationMode | None = ...,
        antialias: bool = ...,
        **kwargs,
    ) -> torch.Tensor: ...
    def center_crop(
        self, image: torch.Tensor, size: SizeDict, **kwargs
    ) -> torch.Tensor: ...

__all__ = ["PoolFormerImageProcessorFast"]
