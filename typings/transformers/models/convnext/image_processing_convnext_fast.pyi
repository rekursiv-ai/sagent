import torch

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...image_utils import ImageInput, PILImageResampling
from ...processing_utils import Unpack
from ...utils import auto_docstring

"""Fast Image processor class for ConvNeXT."""

class ConvNextFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    crop_pct: float | None

@auto_docstring
class ConvNextImageProcessorFast(BaseImageProcessorFast):
    resample = ...
    image_mean = ...
    image_std = ...
    size = ...
    default_to_square = ...
    do_resize = ...
    do_rescale = ...
    do_normalize = ...
    crop_pct = ...
    valid_kwargs = ConvNextFastImageProcessorKwargs
    def __init__(self, **kwargs: Unpack[ConvNextFastImageProcessorKwargs]) -> None: ...
    @auto_docstring
    def preprocess(
        self, images: ImageInput, **kwargs: Unpack[ConvNextFastImageProcessorKwargs]
    ) -> BatchFeature: ...
    def resize(
        self,
        image: torch.Tensor,
        size: dict[str, int],
        crop_pct: float,
        interpolation: PILImageResampling = ...,
        **kwargs,
    ) -> torch.Tensor: ...

__all__ = ["ConvNextImageProcessorFast"]
