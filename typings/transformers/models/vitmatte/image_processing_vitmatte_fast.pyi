import torch

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...processing_utils import Unpack
from ...utils import auto_docstring

"""Fast Image processor class for ViTMatte."""
logger = ...

class VitMatteFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    size_divisor: int | None

@auto_docstring
class VitMatteImageProcessorFast(BaseImageProcessorFast):
    do_rescale: bool = ...
    rescale_factor: int | float = ...
    do_normalize: bool = ...
    image_mean: float | list[float] | None = ...
    image_std: float | list[float] | None = ...
    do_pad: bool = ...
    size_divisor: int = ...
    valid_kwargs = VitMatteFastImageProcessorKwargs
    def __init__(self, **kwargs: Unpack[VitMatteFastImageProcessorKwargs]) -> None: ...
    @property
    def size_divisibility(self):  # -> int:
        ...
    @size_divisibility.setter
    def size_divisibility(self, value):  # -> None:
        ...
    @auto_docstring
    def preprocess(
        self,
        images: list[torch.Tensor],
        trimaps: list[torch.Tensor],
        **kwargs: Unpack[VitMatteFastImageProcessorKwargs],
    ) -> BatchFeature: ...

__all__ = ["VitMatteImageProcessorFast"]
