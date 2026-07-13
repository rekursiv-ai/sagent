from .image_processing_imagegpt import ImageGPTImageProcessor
from ...utils.import_utils import requires

"""Feature extractor class for ImageGPT."""
logger = ...

@requires(backends=("vision",))
class ImageGPTFeatureExtractor(ImageGPTImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["ImageGPTFeatureExtractor"]
