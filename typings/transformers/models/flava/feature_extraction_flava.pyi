from .image_processing_flava import FlavaImageProcessor
from ...utils.import_utils import requires

"""Feature extractor class for FLAVA."""
logger = ...

@requires(backends=("vision",))
class FlavaFeatureExtractor(FlavaImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["FlavaFeatureExtractor"]
