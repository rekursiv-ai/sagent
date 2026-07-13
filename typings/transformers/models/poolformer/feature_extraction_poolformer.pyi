from .image_processing_poolformer import PoolFormerImageProcessor
from ...utils.import_utils import requires

"""Feature extractor class for PoolFormer."""
logger = ...

@requires(backends=("vision",))
class PoolFormerFeatureExtractor(PoolFormerImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["PoolFormerFeatureExtractor"]
