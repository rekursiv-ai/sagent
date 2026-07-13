from torchvision.transforms.v2 import functional as F

import torch

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...image_utils import ImageInput, SizeDict
from ...modeling_outputs import DepthEstimatorOutput
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring

"""Fast Image processor class for PromptDepthAnything."""

class PromptDepthAnythingFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    keep_aspect_ratio: bool | None
    ensure_multiple_of: int | None
    do_pad: bool | None
    size_divisor: int | None
    prompt_scale_to_meter: float | None

@auto_docstring
class PromptDepthAnythingImageProcessorFast(BaseImageProcessorFast):
    model_input_names = ...
    resample = ...
    image_mean = ...
    image_std = ...
    size = ...
    do_resize = ...
    do_rescale = ...
    do_normalize = ...
    keep_aspect_ratio = ...
    ensure_multiple_of = ...
    do_pad = ...
    size_divisor = ...
    prompt_scale_to_meter = ...
    valid_kwargs = PromptDepthAnythingFastImageProcessorKwargs
    def __init__(
        self, **kwargs: Unpack[PromptDepthAnythingFastImageProcessorKwargs]
    ) -> None: ...
    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        prompt_depth: ImageInput | None = ...,
        **kwargs: Unpack[PromptDepthAnythingFastImageProcessorKwargs],
    ) -> BatchFeature: ...
    def resize_with_aspect_ratio(
        self,
        image: torch.Tensor,
        size: SizeDict,
        keep_aspect_ratio: bool = ...,
        ensure_multiple_of: int = ...,
        interpolation: F.InterpolationMode | None = ...,
    ) -> torch.Tensor: ...
    def pad_image(self, image: torch.Tensor, size_divisor: int) -> torch.Tensor: ...
    def post_process_depth_estimation(
        self,
        outputs: DepthEstimatorOutput,
        target_sizes: TensorType | list[tuple[int, int]] | None = ...,
    ) -> list[dict[str, TensorType]]: ...

__all__ = ["PromptDepthAnythingImageProcessorFast"]
