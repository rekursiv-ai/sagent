import numpy as np
import PIL

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType, filter_out_non_signature_kwargs
from ...utils.import_utils import requires

"""Image processor class for ImageGPT."""
logger = ...

def squared_euclidean_distance(a, b):  # -> Any:
    ...
def color_quantize(x, clusters):  # -> Any:
    ...

@requires(backends=("vision",))
class ImageGPTImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        clusters: list[list[int]] | np.ndarray | None = ...,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        do_normalize: bool = ...,
        do_color_quantize: bool = ...,
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
    def normalize(
        self,
        image: np.ndarray,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> np.ndarray: ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling | None = ...,
        do_normalize: bool | None = ...,
        do_color_quantize: bool | None = ...,
        clusters: list[list[int]] | np.ndarray | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> PIL.Image.Image: ...
    def to_dict(self):  # -> dict[str, Any]:
        ...

__all__ = ["ImageGPTImageProcessor"]
