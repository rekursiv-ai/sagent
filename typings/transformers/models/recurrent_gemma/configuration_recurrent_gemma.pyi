from ...configuration_utils import PretrainedConfig

"""RecurrentGemma model configuration"""
logger = ...

class RecurrentGemmaConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        num_hidden_layers=...,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_attention_heads=...,
        lru_width=...,
        attention_window_size=...,
        conv1d_width=...,
        logits_soft_cap=...,
        rms_norm_eps=...,
        use_cache=...,
        pad_token_id=...,
        eos_token_id=...,
        bos_token_id=...,
        hidden_activation=...,
        partial_rotary_factor=...,
        rope_theta=...,
        block_types=...,
        attention_dropout=...,
        num_key_value_heads=...,
        attention_bias=...,
        w_init_variance_scale=...,
        **kwargs,
    ) -> None: ...
    @property
    def layers_block_type(self):  # -> list[Any]:
        ...

__all__ = ["RecurrentGemmaConfig"]
