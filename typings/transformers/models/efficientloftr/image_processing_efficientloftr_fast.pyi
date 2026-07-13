from PIL import Image

import torch

from .modeling_efficientloftr import KeypointMatchingOutput
from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...image_utils import ImageInput
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring

"""Fast Image processor class for EfficientLoFTR."""

def flatten_pair_images(images):  # -> list[Any]:
    ...
def is_grayscale(image: torch.Tensor):  # -> Tensor | Literal[True]:
    ...
def convert_to_grayscale(image: torch.Tensor) -> torch.Tensor: ...

class EfficientLoFTRFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    do_grayscale: bool | None = ...

@auto_docstring
class EfficientLoFTRImageProcessorFast(BaseImageProcessorFast):
    resample = ...
    size = ...
    default_to_square = ...
    do_resize = ...
    do_rescale = ...
    rescale_factor = ...
    do_normalize = ...
    valid_kwargs = EfficientLoFTRFastImageProcessorKwargs
    def __init__(
        self, **kwargs: Unpack[EfficientLoFTRFastImageProcessorKwargs]
    ) -> None: ...
    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[EfficientLoFTRFastImageProcessorKwargs],
    ) -> BatchFeature: ...
    def post_process_keypoint_matching(
        self,
        outputs: KeypointMatchingOutput,
        target_sizes: TensorType | list[tuple],
        threshold: float = ...,
    ) -> list[dict[str, torch.Tensor]]: ...
    def visualize_keypoint_matching(
        self, images, keypoint_matching_output: list[dict[str, torch.Tensor]]
    ) -> list[Image.Image]: ...

__all__ = ["EfficientLoFTRImageProcessorFast"]
