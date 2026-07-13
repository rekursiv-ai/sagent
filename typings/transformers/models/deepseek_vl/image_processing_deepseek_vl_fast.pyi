import torch
import torch.nn.functional as F

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...image_utils import SizeDict
from ...processing_utils import Unpack
from ...utils import auto_docstring

class DeepseekVLFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    min_size: int

@auto_docstring
class DeepseekVLImageProcessorFast(BaseImageProcessorFast):
    resample = ...
    image_mean = ...
    image_std = ...
    size = ...
    min_size = ...
    do_resize = ...
    do_rescale = ...
    do_normalize = ...
    do_pad = ...
    valid_kwargs = DeepseekVLFastImageProcessorKwargs
    def __init__(
        self, **kwargs: Unpack[DeepseekVLFastImageProcessorKwargs]
    ) -> None: ...
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

__all__ = ["DeepseekVLImageProcessorFast"]
