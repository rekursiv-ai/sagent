from .configuration_dots1 import Dots1Config
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3DecoderLayer,
    DeepseekV3MLP,
    DeepseekV3MoE,
    DeepseekV3PreTrainedModel,
    DeepseekV3TopkRouter,
)
from ..qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    TransformersKwargs,
)
from ...modeling_outputs import CausalLMOutputWithPast
from ...processing_utils import Unpack

logger = ...

class Dots1RMSNorm(Qwen3RMSNorm): ...
class Dots1RotaryEmbedding(Qwen3RotaryEmbedding): ...
class Dots1Attention(Qwen3Attention): ...
class Dots1MLP(DeepseekV3MLP): ...
class Dots1MoE(DeepseekV3MoE): ...
class Dots1TopkRouter(DeepseekV3TopkRouter): ...

class Dots1DecoderLayer(DeepseekV3DecoderLayer):
    def __init__(self, config: Dots1Config, layer_idx: int) -> None: ...

class Dots1PreTrainedModel(DeepseekV3PreTrainedModel): ...
class Dots1Model(Qwen3Model): ...

class Dots1ForCausalLM(Qwen3ForCausalLM):
    def forward(
        self, **super_kwargs: Unpack[TransformersKwargs]
    ) -> CausalLMOutputWithPast: ...

__all__ = ["Dots1ForCausalLM", "Dots1Model", "Dots1PreTrainedModel"]
