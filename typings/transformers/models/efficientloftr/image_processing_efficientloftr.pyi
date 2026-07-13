from PIL import Image

import numpy as np
import torch

from .modeling_efficientloftr import KeypointMatchingOutput
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType

"""Image processor class for SuperPoint."""
logger = ...

def is_grayscale(
    image: np.ndarray, input_data_format: str | ChannelDimension | None = ...
):  # -> bool | Literal[True] | None:
    ...
def convert_to_grayscale(
    image: ImageInput, input_data_format: str | ChannelDimension | None = ...
) -> ImageInput: ...
def validate_and_format_image_pairs(
    images: ImageInput,
):  # -> list[Image] | list[ndarray[Any, Any]] | list[Tensor] | list[Any]:
    ...

class EfficientLoFTRImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_grayscale: bool = ...,
        **kwargs,
    ) -> None: ...
    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ):  # -> ndarray[Any, Any]:
        ...
    def preprocess(
        self,
        images,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_grayscale: bool | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> BatchFeature: ...
    def post_process_keypoint_matching(
        self,
        outputs: KeypointMatchingOutput,
        target_sizes: TensorType | list[tuple],
        threshold: float = ...,
    ) -> list[dict[str, torch.Tensor]]: ...
    def visualize_keypoint_matching(
        self,
        images: ImageInput,
        keypoint_matching_output: list[dict[str, torch.Tensor]],
    ) -> list[Image.Image]: ...

__all__ = ["EfficientLoFTRImageProcessor"]
