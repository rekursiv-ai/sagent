from .image_processing_clip import CLIPImageProcessor
from ...utils.import_utils import requires

"""Feature extractor class for CLIP."""
logger = ...

@requires(backends=("vision",))
class CLIPFeatureExtractor(CLIPImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["CLIPFeatureExtractor"]
