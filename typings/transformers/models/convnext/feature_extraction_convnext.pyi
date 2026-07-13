from .image_processing_convnext import ConvNextImageProcessor
from ...utils.import_utils import requires

"""Feature extractor class for ConvNeXT."""
logger = ...

@requires(backends=("vision",))
class ConvNextFeatureExtractor(ConvNextImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["ConvNextFeatureExtractor"]
