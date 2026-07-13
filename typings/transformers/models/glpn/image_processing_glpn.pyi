import numpy as np
import PIL.Image

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_utils import ChannelDimension, PILImageResampling
from ...modeling_outputs import DepthEstimatorOutput
from ...utils import TensorType, filter_out_non_signature_kwargs
from ...utils.import_utils import requires

"""Image processor class for GLPN."""
logger = ...

@requires(backends=("vision",))
class GLPNImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_resize: bool = ...,
        size_divisor: int = ...,
        resample=...,
        do_rescale: bool = ...,
        **kwargs,
    ) -> None: ...
    def resize(
        self,
        image: np.ndarray,
        size_divisor: int,
        resample: PILImageResampling = ...,
        data_format: ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: PIL.Image.Image | TensorType | list[PIL.Image.Image] | list[TensorType],
        do_resize: bool | None = ...,
        size_divisor: int | None = ...,
        resample=...,
        do_rescale: bool | None = ...,
        return_tensors: TensorType | str | None = ...,
        data_format: ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> BatchFeature: ...
    def post_process_depth_estimation(
        self,
        outputs: DepthEstimatorOutput,
        target_sizes: TensorType | list[tuple[int, int]] | None = ...,
    ) -> list[dict[str, TensorType]]: ...

__all__ = ["GLPNImageProcessor"]
