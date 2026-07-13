from torchvision.transforms.v2 import functional as F

import torch

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    BatchFeature,
    DefaultFastImageProcessorKwargs,
    ImageInput,
    SizeDict,
    Unpack,
)
from ...utils import auto_docstring

"""Fast Image processor class for BridgeTower."""

def make_pixel_mask(
    image: torch.Tensor, output_size: tuple[int, int]
) -> torch.Tensor: ...
def get_resize_output_image_size(
    input_image: torch.Tensor,
    shorter: int = ...,
    longer: int = ...,
    size_divisor: int = ...,
) -> tuple[int, int]: ...

class BridgeTowerFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    size_divisor: int | None

@auto_docstring
class BridgeTowerImageProcessorFast(BaseImageProcessorFast):
    resample = ...
    image_mean = ...
    image_std = ...
    size = ...
    default_to_square = ...
    crop_size = ...
    do_resize = ...
    do_center_crop = ...
    do_rescale = ...
    do_normalize = ...
    do_pad = ...
    size_divisor = ...
    valid_kwargs = BridgeTowerFastImageProcessorKwargs
    model_input_names = ...
    def __init__(
        self, **kwargs: Unpack[BridgeTowerFastImageProcessorKwargs]
    ) -> None: ...
    @auto_docstring
    def preprocess(
        self, images: ImageInput, **kwargs: Unpack[BridgeTowerFastImageProcessorKwargs]
    ) -> BatchFeature: ...
    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        size_divisor: int = ...,
        interpolation: F.InterpolationMode | None = ...,
        antialias: bool = ...,
        **kwargs,
    ) -> torch.Tensor: ...
    def center_crop(
        self, image: torch.Tensor, size: dict[str, int], **kwargs
    ) -> torch.Tensor: ...
    def to_dict(self):  # -> dict[str, Any]:
        ...

__all__ = ["BridgeTowerImageProcessorFast"]
