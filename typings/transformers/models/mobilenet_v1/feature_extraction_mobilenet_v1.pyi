from .image_processing_mobilenet_v1 import MobileNetV1ImageProcessor
from ...utils.import_utils import requires

"""Feature extractor class for MobileNetV1."""
logger = ...

@requires(backends=("vision",))
class MobileNetV1FeatureExtractor(MobileNetV1ImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["MobileNetV1FeatureExtractor"]
