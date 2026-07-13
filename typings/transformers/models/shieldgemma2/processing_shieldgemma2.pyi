from collections.abc import Mapping, Sequence

from ..gemma3.processing_gemma3 import Gemma3Processor, Gemma3ProcessorKwargs
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import Unpack

logger = ...
DEFAULT_SHIELDGEMMA2_POLICIES: Mapping[str, str] = ...

class ShieldGemma2ProcessorKwargs(Gemma3ProcessorKwargs, total=False):
    policies: Sequence[str] | None
    custom_policies: Mapping[str, str] | None
    _defaults = ...

class ShieldGemma2Processor(Gemma3Processor):
    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template=...,
        image_seq_length=...,
        policy_definitions=...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text=...,
        videos=...,
        audio=...,
        **kwargs: Unpack[ShieldGemma2ProcessorKwargs],
    ) -> BatchFeature: ...

__all__ = ["ShieldGemma2Processor"]
