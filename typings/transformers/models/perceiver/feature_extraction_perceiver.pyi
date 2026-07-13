from .image_processing_perceiver import PerceiverImageProcessor
from ...utils.import_utils import requires

"""Feature extractor class for Perceiver."""
logger = ...

@requires(backends=("vision",))
class PerceiverFeatureExtractor(PerceiverImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["PerceiverFeatureExtractor"]
