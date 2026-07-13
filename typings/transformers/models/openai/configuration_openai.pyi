from ...configuration_utils import PretrainedConfig

"""OpenAI GPT configuration"""
logger = ...

class OpenAIGPTConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        n_positions=...,
        n_embd=...,
        n_layer=...,
        n_head=...,
        afn=...,
        resid_pdrop=...,
        embd_pdrop=...,
        attn_pdrop=...,
        layer_norm_epsilon=...,
        initializer_range=...,
        summary_type=...,
        summary_use_proj=...,
        summary_activation=...,
        summary_proj_to_labels=...,
        summary_first_dropout=...,
        **kwargs,
    ) -> None: ...

__all__ = ["OpenAIGPTConfig"]
