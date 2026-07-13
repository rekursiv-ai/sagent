import numpy as np
import torch

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...processing_utils import Unpack
from ...utils import auto_docstring

"""Fast Image processor class for ImageGPT."""

def squared_euclidean_distance_torch(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor: ...
def color_quantize_torch(x: torch.Tensor, clusters: torch.Tensor) -> torch.Tensor: ...

class ImageGPTFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    clusters: np.ndarray | list[list[int]] | torch.Tensor | None
    do_color_quantize: bool | None

@auto_docstring
class ImageGPTImageProcessorFast(BaseImageProcessorFast):
    model_input_names = ...
    resample = ...
    do_color_quantize = ...
    clusters = ...
    image_mean = ...
    image_std = ...
    do_rescale = ...
    do_normalize = ...
    valid_kwargs = ImageGPTFastImageProcessorKwargs
    def __init__(
        self,
        clusters: list | np.ndarray | torch.Tensor | None = ...,
        **kwargs: Unpack[ImageGPTFastImageProcessorKwargs],
    ) -> None: ...
    def to_dict(self):  # -> dict[str, Any]:
        ...

__all__ = ["ImageGPTImageProcessorFast"]
