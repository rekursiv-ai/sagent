from .image_processing_layoutlmv2 import LayoutLMv2ImageProcessor
from ...utils.import_utils import requires

"""
Feature extractor class for LayoutLMv2.
"""
logger = ...

@requires(backends=("vision",))
class LayoutLMv2FeatureExtractor(LayoutLMv2ImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["LayoutLMv2FeatureExtractor"]
