from torchvision.transforms.v2 import functional as F

import torch

from .modeling_zoedepth import ZoeDepthDepthEstimatorOutput
from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...image_utils import ImageInput, SizeDict
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring

"""Fast Image processor class for ZoeDepth."""
logger = ...

class ZoeDepthFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    keep_aspect_ratio: bool | None
    ensure_multiple_of: int | None

@auto_docstring
class ZoeDepthImageProcessorFast(BaseImageProcessorFast):
    do_pad = ...
    do_rescale = ...
    do_normalize = ...
    image_mean = ...
    image_std = ...
    do_resize = ...
    size = ...
    resample = ...
    keep_aspect_ratio = ...
    ensure_multiple_of = ...
    valid_kwargs = ZoeDepthFastImageProcessorKwargs
    def __init__(self, **kwargs: Unpack[ZoeDepthFastImageProcessorKwargs]) -> None: ...
    @auto_docstring
    def preprocess(
        self, images: ImageInput, **kwargs: Unpack[ZoeDepthFastImageProcessorKwargs]
    ) -> BatchFeature: ...
    def resize(
        self,
        images: torch.Tensor,
        size: SizeDict,
        keep_aspect_ratio: bool = ...,
        ensure_multiple_of: int = ...,
        interpolation: F.InterpolationMode | None = ...,
    ) -> torch.Tensor: ...
    def post_process_depth_estimation(
        self,
        outputs: ZoeDepthDepthEstimatorOutput,
        source_sizes: TensorType | list[tuple[int, int]] | None = ...,
        target_sizes: TensorType | list[tuple[int, int]] | None = ...,
        outputs_flipped: ZoeDepthDepthEstimatorOutput | None = ...,
        do_remove_padding: bool | None = ...,
    ) -> list[dict[str, TensorType]]: ...

__all__ = ["ZoeDepthImageProcessorFast"]
