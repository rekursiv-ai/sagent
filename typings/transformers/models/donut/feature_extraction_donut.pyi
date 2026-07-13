from .image_processing_donut import DonutImageProcessor
from ...utils.import_utils import requires

"""Feature extractor class for Donut."""
logger = ...

@requires(backends=("vision",))
class DonutFeatureExtractor(DonutImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["DonutFeatureExtractor"]
