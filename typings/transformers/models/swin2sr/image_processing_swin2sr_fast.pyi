import torch

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...image_utils import ImageInput
from ...processing_utils import Unpack
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""Fast Image processor class for Swin2SR."""
logger = ...

class Swin2SRFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    size_divisor: int | None

@auto_docstring
class Swin2SRImageProcessorFast(BaseImageProcessorFast):
    do_rescale = ...
    rescale_factor = ...
    do_pad = ...
    size_divisor = ...
    valid_kwargs = Swin2SRFastImageProcessorKwargs
    def __init__(self, **kwargs: Unpack[Swin2SRFastImageProcessorKwargs]) -> None: ...
    @property
    def pad_size(self):  # -> int:
        ...
    @pad_size.setter
    def pad_size(self, value):  # -> None:
        ...
    def preprocess(
        self, images: ImageInput, **kwargs: Unpack[Swin2SRFastImageProcessorKwargs]
    ) -> BatchFeature: ...
    @deprecate_kwarg("size", version="v5", new_name="size_divisor")
    def pad(self, images: torch.Tensor, size_divisor: int) -> torch.Tensor: ...

__all__ = ["Swin2SRImageProcessorFast"]
