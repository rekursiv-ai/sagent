from .image_processing_chinese_clip import ChineseCLIPImageProcessor
from ...utils.import_utils import requires

"""Feature extractor class for Chinese-CLIP."""
logger = ...

@requires(backends=("vision",))
class ChineseCLIPFeatureExtractor(ChineseCLIPImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["ChineseCLIPFeatureExtractor"]
