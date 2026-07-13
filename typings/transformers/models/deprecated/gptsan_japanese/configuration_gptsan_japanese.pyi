from ....configuration_utils import PretrainedConfig

"""GPTSAN-japanese model configuration"""
logger = ...

class GPTSanJapaneseConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        max_position_embeddings=...,
        d_model=...,
        d_ff=...,
        d_ext=...,
        d_spout=...,
        num_switch_layers=...,
        num_ext_layers=...,
        num_heads=...,
        num_experts=...,
        expert_capacity=...,
        dropout_rate=...,
        layer_norm_epsilon=...,
        router_bias=...,
        router_jitter_noise=...,
        router_dtype=...,
        router_ignore_padding_tokens=...,
        output_hidden_states=...,
        output_attentions=...,
        initializer_factor=...,
        output_router_logits=...,
        use_cache=...,
        separator_token_id=...,
        pad_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["GPTSanJapaneseConfig"]
