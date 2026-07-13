from ...configuration_utils import PretrainedConfig

"""JetMoe model configuration"""
logger = ...

class JetMoeConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_key_value_heads=...,
        kv_channels=...,
        intermediate_size=...,
        max_position_embeddings=...,
        activation_function=...,
        num_local_experts=...,
        num_experts_per_tok=...,
        output_router_logits=...,
        aux_loss_coef=...,
        use_cache=...,
        bos_token_id=...,
        eos_token_id=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rms_norm_eps=...,
        initializer_range=...,
        attention_dropout=...,
        **kwargs,
    ) -> None: ...

__all__ = ["JetMoeConfig"]
