import numpy as np

from ....image_processing_utils import BaseImageProcessor, BatchFeature
from ....image_utils import ChannelDimension, ImageInput, PILImageResampling
from ....utils import TensorType

"""Image processor class for TVLT."""
logger = ...

def make_batched(videos) -> list[list[ImageInput]]: ...

class TvltImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        patch_size: list[int] = ...,
        num_frames: int = ...,
        resample: PILImageResampling = ...,
        do_center_crop: bool = ...,
        crop_size: dict[str, int] | None = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        init_mask_generator=...,
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
    def preprocess(
        self,
        videos: ImageInput,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        patch_size: list[int] | None = ...,
        num_frames: int | None = ...,
        resample: PILImageResampling | None = ...,
        do_center_crop: bool | None = ...,
        crop_size: dict[str, int] | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        is_mixed: bool = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> BatchFeature: ...

__all__ = ["TvltImageProcessor"]
