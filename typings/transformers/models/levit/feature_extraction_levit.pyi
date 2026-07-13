from .image_processing_levit import LevitImageProcessor
from ...utils.import_utils import requires

"""Feature extractor class for LeViT."""
logger = ...

@requires(backends=("vision",))
class LevitFeatureExtractor(LevitImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["LevitFeatureExtractor"]
