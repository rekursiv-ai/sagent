from PIL import Image

import numpy as np

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput
from ...utils import TensorType

"""Image processor class for Pix2Struct."""
logger = ...
DEFAULT_FONT_PATH = ...

def torch_extract_patches(image_tensor, patch_height, patch_width):  # -> Tensor:
    ...
def render_text(
    text: str,
    text_size: int = ...,
    text_color: str = ...,
    background_color: str = ...,
    left_padding: int = ...,
    right_padding: int = ...,
    top_padding: int = ...,
    bottom_padding: int = ...,
    font_bytes: bytes | None = ...,
    font_path: str | None = ...,
) -> Image.Image: ...
def render_header(
    image: np.ndarray,
    header: str,
    input_data_format: str | ChildProcessError | None = ...,
    **kwargs,
):  # -> ndarray[Any, Any]:
    ...

class Pix2StructImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_convert_rgb: bool = ...,
        do_normalize: bool = ...,
        patch_size: dict[str, int] | None = ...,
        max_patches: int = ...,
        is_vqa: bool = ...,
        **kwargs,
    ) -> None: ...
    def extract_flattened_patches(
        self,
        image: np.ndarray,
        max_patches: int,
        patch_size: dict,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    def normalize(
        self,
        image: np.ndarray,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    def preprocess(
        self,
        images: ImageInput,
        header_text: str | None = ...,
        do_convert_rgb: bool | None = ...,
        do_normalize: bool | None = ...,
        max_patches: int | None = ...,
        patch_size: dict[str, int] | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> ImageInput: ...

__all__ = ["Pix2StructImageProcessor"]
