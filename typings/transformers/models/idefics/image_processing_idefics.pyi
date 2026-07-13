from collections.abc import Callable

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ImageInput
from ...utils import TensorType

"""Image processor class for Idefics."""
IDEFICS_STANDARD_MEAN = ...
IDEFICS_STANDARD_STD = ...

def convert_to_rgb(image):  # -> Image:
    ...

class IdeficsImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        image_size: int = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        image_num_channels: int | None = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        **kwargs,
    ) -> None: ...
    def preprocess(
        self,
        images: ImageInput,
        image_num_channels: int | None = ...,
        image_size: dict[str, int] | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        transform: Callable | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        return_tensors: str | TensorType | None = ...,
        **kwargs,
    ) -> TensorType: ...

__all__ = ["IdeficsImageProcessor"]
