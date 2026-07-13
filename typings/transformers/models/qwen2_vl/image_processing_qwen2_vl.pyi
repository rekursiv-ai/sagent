from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType
from ...video_utils import VideoInput

"""Image processor class for Qwen2-VL."""
logger = ...

def smart_resize(
    height: int,
    width: int,
    factor: int = ...,
    min_pixels: int = ...,
    max_pixels: int = ...,
):  # -> tuple[int, int]:
    ...

class Qwen2VLImageProcessor(BaseImageProcessor):
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
        min_pixels: int | None = ...,
        max_pixels: int | None = ...,
        patch_size: int = ...,
        temporal_patch_size: int = ...,
        merge_size: int = ...,
        **kwargs,
    ) -> None: ...
    def preprocess(
        self,
        images: ImageInput,
        videos: VideoInput | None = ...,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        min_pixels: int | None = ...,
        max_pixels: int | None = ...,
        resample: PILImageResampling | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        patch_size: int | None = ...,
        temporal_patch_size: int | None = ...,
        merge_size: int | None = ...,
        do_convert_rgb: bool | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> BatchFeature:
        ...
    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs=...
    ): ...

__all__ = ["Qwen2VLImageProcessor"]
