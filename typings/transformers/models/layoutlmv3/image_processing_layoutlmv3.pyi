from collections.abc import Iterable

import numpy as np
import PIL

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import (
    TensorType,
    filter_out_non_signature_kwargs,
)
from ...utils.import_utils import requires

"""Image processor class for LayoutLMv3."""
logger = ...

def normalize_box(box, width, height):  # -> list[int]:
    ...
def apply_tesseract(
    image: np.ndarray,
    lang: str | None,
    tesseract_config: str | None,
    input_data_format: ChannelDimension | str | None = ...,
):  # -> tuple[list[Any], list[Any]]:
    ...

@requires(backends=("vision",))
class LayoutLMv3ImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        do_rescale: bool = ...,
        rescale_value: float = ...,
        do_normalize: bool = ...,
        image_mean: float | Iterable[float] | None = ...,
        image_std: float | Iterable[float] | None = ...,
        apply_ocr: bool = ...,
        ocr_lang: str | None = ...,
        tesseract_config: str | None = ...,
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
        resample=...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | Iterable[float] | None = ...,
        image_std: float | Iterable[float] | None = ...,
        apply_ocr: bool | None = ...,
        ocr_lang: str | None = ...,
        tesseract_config: str | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> PIL.Image.Image: ...

__all__ = ["LayoutLMv3ImageProcessor"]
