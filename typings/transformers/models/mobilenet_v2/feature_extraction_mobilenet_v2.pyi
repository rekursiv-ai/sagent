from .image_processing_mobilenet_v2 import MobileNetV2ImageProcessor
from ...utils.import_utils import requires

"""Feature extractor class for MobileNetV2."""
logger = ...

@requires(backends=("vision",))
class MobileNetV2FeatureExtractor(MobileNetV2ImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["MobileNetV2FeatureExtractor"]
