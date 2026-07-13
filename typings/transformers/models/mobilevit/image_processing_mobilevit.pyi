import numpy as np
import PIL

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import (
    TensorType,
    filter_out_non_signature_kwargs,
)
from ...utils.import_utils import requires

"""Image processor class for MobileViT."""
logger = ...

@requires(backends=("vision",))
class MobileViTImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_center_crop: bool = ...,
        crop_size: dict[str, int] | None = ...,
        do_flip_channel_order: bool = ...,
        do_reduce_labels: bool = ...,
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
    def flip_channel_order(
        self,
        image: np.ndarray,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> np.ndarray: ...
    def reduce_label(self, label: ImageInput) -> np.ndarray: ...
    def __call__(self, images, segmentation_maps=..., **kwargs):  # -> BatchFeature:
        ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None = ...,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_center_crop: bool | None = ...,
        crop_size: dict[str, int] | None = ...,
        do_flip_channel_order: bool | None = ...,
        do_reduce_labels: bool | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> PIL.Image.Image: ...
    def post_process_semantic_segmentation(
        self, outputs, target_sizes: list[tuple] | None = ...
    ):  # -> list[Any]:
        ...

__all__ = ["MobileViTImageProcessor"]
