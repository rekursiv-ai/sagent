import numpy as np

from .modeling_depth_pro import DepthProDepthEstimatorOutput
from ...image_processing_utils import BaseImageProcessor
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
)
from ...utils import (
    TensorType,
    filter_out_non_signature_kwargs,
)
from ...utils.import_utils import requires

"""Image processor class for DepthPro."""
logger = ...

@requires(backends=("torchvision", "torch"))
class DepthProImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        **kwargs,
    ) -> None: ...
    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        resample: PILImageResampling = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: str | ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> BatchFeature:
        ...
    def post_process_depth_estimation(
        self,
        outputs: DepthProDepthEstimatorOutput,
        target_sizes: TensorType | list[tuple[int, int]] | None = ...,
    ) -> list[dict[str, TensorType]]: ...

__all__ = ["DepthProImageProcessor"]
