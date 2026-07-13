from torchvision.transforms.v2 import functional as F

import torch

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...image_utils import ImageInput, SizeDict
from ...processing_utils import Unpack
from ...utils import auto_docstring

class JanusFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    min_size: int

@auto_docstring
class JanusImageProcessorFast(BaseImageProcessorFast):
    resample = ...
    image_mean = ...
    image_std = ...
    size = ...
    min_size = ...
    do_resize = ...
    do_rescale = ...
    do_normalize = ...
    do_pad = ...
    valid_kwargs = JanusFastImageProcessorKwargs
    def __init__(self, **kwargs: Unpack[JanusFastImageProcessorKwargs]) -> None: ...
    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        min_size: int,
        interpolation: F.InterpolationMode | None = ...,
        antialias: bool = ...,
        **kwargs,
    ) -> torch.Tensor: ...
    def pad_to_square(
        self,
        images: torch.Tensor,
        background_color: int | tuple[int, int, int] = ...,
    ) -> torch.Tensor: ...
    def postprocess(
        self,
        images: ImageInput,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: list[float] | None = ...,
        image_std: list[float] | None = ...,
        return_tensors: str | None = ...,
    ) -> torch.Tensor: ...

__all__ = ["JanusImageProcessorFast"]
