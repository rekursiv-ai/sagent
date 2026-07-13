import torch

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...image_utils import ImageInput
from ...processing_utils import Unpack
from ...utils import auto_docstring

"""Fast Image processor class for Kosmos2_5."""

def torch_extract_patches(image_tensor, patch_height, patch_width):  # -> Tensor:
    ...

class Kosmos2_5FastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    patch_size: dict[str, int] | None
    max_patches: int | None

@auto_docstring
class Kosmos2_5ImageProcessorFast(BaseImageProcessorFast):
    do_normalize = ...
    do_convert_rgb = ...
    patch_size = ...
    max_patches = ...
    rescale_factor = ...
    valid_kwargs = Kosmos2_5FastImageProcessorKwargs
    def __init__(self, **kwargs: Unpack[Kosmos2_5FastImageProcessorKwargs]) -> None: ...
    @auto_docstring
    def preprocess(
        self, images: ImageInput, **kwargs: Unpack[Kosmos2_5FastImageProcessorKwargs]
    ) -> BatchFeature: ...
    def normalize(self, image: torch.Tensor, **kwargs) -> torch.Tensor: ...
    def extract_flattened_patches(
        self, image: torch.Tensor, max_patches: int, patch_size: dict
    ) -> torch.Tensor: ...

__all__ = ["Kosmos2_5ImageProcessorFast"]
