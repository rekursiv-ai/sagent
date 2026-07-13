from .image_processing_maskformer import MaskFormerImageProcessor
from ...utils.import_utils import requires

"""Feature extractor class for MaskFormer."""
logger = ...

@requires(backends=("vision",))
class MaskFormerFeatureExtractor(MaskFormerImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["MaskFormerFeatureExtractor"]
