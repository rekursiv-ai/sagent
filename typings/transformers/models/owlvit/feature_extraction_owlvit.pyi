from .image_processing_owlvit import OwlViTImageProcessor
from ...utils.import_utils import requires

"""Feature extractor class for OwlViT."""
logger = ...

@requires(backends=("vision",))
class OwlViTFeatureExtractor(OwlViTImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["OwlViTFeatureExtractor"]
