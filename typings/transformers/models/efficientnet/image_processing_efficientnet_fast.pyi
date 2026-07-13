import torch

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    BatchFeature,
    DefaultFastImageProcessorKwargs,
)
from ...image_utils import ImageInput
from ...processing_utils import Unpack
from ...utils import auto_docstring

"""Fast Image processor class for EfficientNet."""

class EfficientNetFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    rescale_offset: bool
    include_top: bool

@auto_docstring
class EfficientNetImageProcessorFast(BaseImageProcessorFast):
    resample = ...
    image_mean = ...
    image_std = ...
    size = ...
    crop_size = ...
    do_resize = ...
    do_center_crop = ...
    do_rescale = ...
    rescale_factor = ...
    rescale_offset = ...
    do_normalize = ...
    include_top = ...
    valid_kwargs = EfficientNetFastImageProcessorKwargs
    def __init__(
        self, **kwargs: Unpack[EfficientNetFastImageProcessorKwargs]
    ) -> None: ...
    def rescale(
        self, image: torch.Tensor, scale: float, offset: bool | None = ..., **kwargs
    ) -> torch.Tensor: ...
    def rescale_and_normalize(
        self,
        images: torch.Tensor,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float],
        image_std: float | list[float],
        rescale_offset: bool = ...,
    ) -> torch.Tensor: ...
    @auto_docstring
    def preprocess(
        self, images: ImageInput, **kwargs: Unpack[EfficientNetFastImageProcessorKwargs]
    ) -> BatchFeature: ...

__all__ = ["EfficientNetImageProcessorFast"]
