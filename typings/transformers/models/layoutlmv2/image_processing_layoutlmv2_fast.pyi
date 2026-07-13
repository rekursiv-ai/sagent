from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    BatchFeature,
    DefaultFastImageProcessorKwargs,
)
from ...image_utils import ImageInput
from ...processing_utils import Unpack
from ...utils import auto_docstring

"""Fast Image processor class for LayoutLMv2."""
logger = ...

class LayoutLMv2FastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    apply_ocr: bool | None
    ocr_lang: str | None
    tesseract_config: str | None

@auto_docstring
class LayoutLMv2ImageProcessorFast(BaseImageProcessorFast):
    resample = ...
    size = ...
    rescale_factor = ...
    do_resize = ...
    apply_ocr = ...
    ocr_lang = ...
    tesseract_config = ...
    valid_kwargs = LayoutLMv2FastImageProcessorKwargs
    def __init__(
        self, **kwargs: Unpack[LayoutLMv2FastImageProcessorKwargs]
    ) -> None: ...
    @auto_docstring
    def preprocess(
        self, images: ImageInput, **kwargs: Unpack[LayoutLMv2FastImageProcessorKwargs]
    ) -> BatchFeature: ...

__all__ = ["LayoutLMv2ImageProcessorFast"]
