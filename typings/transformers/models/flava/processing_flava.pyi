from collections.abc import Iterable

from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin

"""
Image/Text processor class for FLAVA
"""

class FlavaImagesKwargs(ImagesKwargs):
    return_image_mask: bool | None
    input_size_patches: int | None
    total_mask_patches: int | None
    mask_group_min_patches: int | None
    mask_group_max_patches: int | None
    mask_group_min_aspect_ratio: float | None
    mask_group_max_aspect_ratio: float | None
    return_codebook_pixels: bool | None
    codebook_do_resize: bool | None
    codebook_size: bool | None
    codebook_resample: int | None
    codebook_do_center_crop: bool | None
    codebook_crop_size: int | None
    codebook_do_rescale: bool | None
    codebook_rescale_factor: int | float | None
    codebook_do_map_pixels: bool | None
    codebook_do_normalize: bool | None
    codebook_image_mean: float | Iterable[float] | None
    codebook_image_std: float | Iterable[float] | None

class FlavaProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: FlavaImagesKwargs
    _defaults = ...

class FlavaProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    valid_processor_kwargs = FlavaProcessorKwargs
    def __init__(self, image_processor=..., tokenizer=..., **kwargs) -> None: ...
    @property
    def feature_extractor_class(self):  # -> str:
        ...
    @property
    def feature_extractor(self): ...

__all__ = ["FlavaProcessor"]
