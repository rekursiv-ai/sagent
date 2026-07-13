from .image_processing_layoutlmv3 import LayoutLMv3ImageProcessor
from ...utils.import_utils import requires

"""
Feature extractor class for LayoutLMv3.
"""
logger = ...

@requires(backends=("vision",))
class LayoutLMv3FeatureExtractor(LayoutLMv3ImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["LayoutLMv3FeatureExtractor"]
