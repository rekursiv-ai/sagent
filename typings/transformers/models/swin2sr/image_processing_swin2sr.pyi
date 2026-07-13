import numpy as np

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput
from ...utils import TensorType, filter_out_non_signature_kwargs
from ...utils.deprecation import deprecate_kwarg

"""Image processor class for Swin2SR."""
logger = ...

class Swin2SRImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_pad: bool = ...,
        size_divisor: int = ...,
        **kwargs,
    ) -> None: ...
    @property
    def pad_size(self):  # -> int:
        ...
    @pad_size.setter
    def pad_size(self, value):  # -> None:
        ...
    def pad(
        self,
        image: np.ndarray,
        size: int,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> ndarray[Any, Any]:
        ...
    @filter_out_non_signature_kwargs()
    @deprecate_kwarg("pad_size", version="v5", new_name="size_divisor")
    def preprocess(
        self,
        images: ImageInput,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_pad: bool | None = ...,
        size_divisor: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: str | ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> BatchFeature:
        ...

__all__ = ["Swin2SRImageProcessor"]
