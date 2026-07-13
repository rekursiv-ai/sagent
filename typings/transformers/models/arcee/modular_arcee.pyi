from transformers.utils import auto_docstring

from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
)
from ..nemotron.modeling_nemotron import NemotronMLP

"""PyTorch Arcee model."""
logger = ...

class ArceeConfig(LlamaConfig):
    model_type = ...
    base_model_tp_plan = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        attention_bias=...,
        attention_dropout=...,
        mlp_bias=...,
        head_dim=...,
        **kwargs,
    ) -> None: ...

class ArceeMLP(NemotronMLP): ...

@auto_docstring(checkpoint="arcee-ai/AFM-4.5B")
class ArceeForCausalLM(LlamaForCausalLM): ...

@auto_docstring(checkpoint="arcee-ai/AFM-4.5B")
class ArceeForSequenceClassification(LlamaForSequenceClassification): ...

@auto_docstring(checkpoint="arcee-ai/AFM-4.5B")
class ArceeForQuestionAnswering(LlamaForQuestionAnswering): ...

@auto_docstring(checkpoint="arcee-ai/AFM-4.5B")
class ArceeForTokenClassification(LlamaForTokenClassification): ...

__all__ = [
    "ArceeConfig",
    "ArceeForCausalLM",
    "ArceeForQuestionAnswering",
    "ArceeForSequenceClassification",
    "ArceeForTokenClassification",
    "ArceeModel",
    "ArceePreTrainedModel",
]
