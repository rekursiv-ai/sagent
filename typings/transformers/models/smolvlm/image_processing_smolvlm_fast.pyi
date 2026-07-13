from torchvision.transforms import functional as F

import torch

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    SizeDict,
)
from ...image_utils import ImageInput
from ...processing_utils import Unpack
from ...utils import auto_docstring

logger = ...

class SmolVLMFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    do_image_splitting: bool | None
    max_image_size: dict[str, int] | None
    return_row_col_info: bool | None

MAX_IMAGE_SIZE = ...

def get_resize_output_image_size(
    image, resolution_max_side: int
) -> tuple[int, int]: ...
def get_max_height_width(images_list: list[list[torch.Tensor]]) -> tuple[int, int]: ...

@auto_docstring
class SmolVLMImageProcessorFast(BaseImageProcessorFast):
    resample = ...
    image_mean = ...
    image_std = ...
    size = ...
    max_image_size = ...
    do_resize = ...
    do_rescale = ...
    do_normalize = ...
    do_convert_rgb = ...
    do_image_splitting = ...
    do_pad = ...
    return_row_col_info = ...
    valid_kwargs = SmolVLMFastImageProcessorKwargs
    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        interpolation: F.InterpolationMode | None = ...,
        antialias: bool = ...,
        **kwargs,
    ) -> torch.Tensor: ...
    def split_images(
        self,
        images: torch.Tensor,
        max_image_size: dict[str, int],
        interpolation: F.InterpolationMode | None = ...,
    ):  # -> tuple[Tensor, list[int], list[int]]:
        ...
    def resize_for_vision_encoder(
        self,
        image: torch.Tensor,
        vision_encoder_max_size: int,
        interpolation: F.InterpolationMode | None = ...,
    ):  # -> Tensor:
        ...
    def pad(
        self,
        image: torch.Tensor,
        padded_size: tuple[int, int],
        fill: int = ...,
        return_pixel_mask: bool = ...,
    ):  # -> tuple[Tensor, Tensor | None]:
        ...
    @auto_docstring
    def preprocess(
        self, images: ImageInput, **kwargs: Unpack[SmolVLMFastImageProcessorKwargs]
    ) -> BatchFeature: ...
    def to_dict(self):  # -> dict[str, Any]:
        ...
    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs=...
    ):  # -> tuple[Any | Literal[1], Any | Literal[1], Any | Literal[1]]:
        ...

__all__ = ["SmolVLMImageProcessorFast"]
