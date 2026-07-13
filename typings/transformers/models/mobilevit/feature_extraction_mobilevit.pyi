from .image_processing_mobilevit import MobileViTImageProcessor
from ...utils.import_utils import requires

"""Feature extractor class for MobileViT."""
logger = ...

@requires(backends=("vision",))
class MobileViTFeatureExtractor(MobileViTImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["MobileViTFeatureExtractor"]
