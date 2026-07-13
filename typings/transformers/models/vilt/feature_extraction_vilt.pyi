from .image_processing_vilt import ViltImageProcessor
from ...utils.import_utils import requires

"""Feature extractor class for ViLT."""
logger = ...

@requires(backends=("vision",))
class ViltFeatureExtractor(ViltImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["ViltFeatureExtractor"]
