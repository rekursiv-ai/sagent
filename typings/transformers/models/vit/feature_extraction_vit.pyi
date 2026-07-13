from .image_processing_vit import ViTImageProcessor
from ...utils.import_utils import requires

"""Feature extractor class for ViT."""
logger = ...

@requires(backends=("vision",))
class ViTFeatureExtractor(ViTImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["ViTFeatureExtractor"]
