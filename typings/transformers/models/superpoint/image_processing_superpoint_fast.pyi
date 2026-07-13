import torch

from .modeling_superpoint import SuperPointKeypointDescriptionOutput
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring

"""Fast Image processor class for Superpoint."""

def is_grayscale(image: torch.Tensor):  # -> Tensor | Literal[True]:
    ...

class SuperPointFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    do_grayscale: bool | None = ...

def convert_to_grayscale(image: torch.Tensor) -> torch.Tensor: ...

@auto_docstring
class SuperPointImageProcessorFast(BaseImageProcessorFast):
    resample = ...
    size = ...
    default_to_square = ...
    do_resize = ...
    do_rescale = ...
    rescale_factor = ...
    do_normalize = ...
    valid_kwargs = SuperPointFastImageProcessorKwargs
    def __init__(
        self, **kwargs: Unpack[SuperPointFastImageProcessorKwargs]
    ) -> None: ...
    def post_process_keypoint_detection(
        self,
        outputs: SuperPointKeypointDescriptionOutput,
        target_sizes: TensorType | list[tuple],
    ) -> list[dict[str, torch.Tensor]]: ...

__all__ = ["SuperPointImageProcessorFast"]
