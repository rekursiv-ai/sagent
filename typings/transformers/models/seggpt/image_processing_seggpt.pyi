import numpy as np

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType

"""Image processor class for SegGPT."""
logger = ...

def build_palette(num_labels: int) -> list[tuple[int, int]]: ...
def mask_to_rgb(
    mask: np.ndarray,
    palette: list[tuple[int, int]] | None = ...,
    data_format: ChannelDimension | None = ...,
) -> np.ndarray: ...

class SegGptImageProcessor(BaseImageProcessor):
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
        do_convert_rgb: bool = ...,
        **kwargs,
    ) -> None: ...
    def get_palette(self, num_labels: int) -> list[tuple[int, int]]: ...
    def mask_to_rgb(
        self,
        image: np.ndarray,
        palette: list[tuple[int, int]] | None = ...,
        data_format: str | ChannelDimension | None = ...,
    ) -> np.ndarray: ...
    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        resample: PILImageResampling = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    def preprocess(
        self,
        images: ImageInput | None = ...,
        prompt_images: ImageInput | None = ...,
        prompt_masks: ImageInput | None = ...,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_convert_rgb: bool | None = ...,
        num_labels: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: str | ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ):  # -> BatchFeature:
        ...
    def post_process_semantic_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]] | None = ...,
        num_labels: int | None = ...,
    ):  # -> list[Any]:
        ...

__all__ = ["SegGptImageProcessor"]
