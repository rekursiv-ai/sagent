from ...configuration_utils import PretrainedConfig

"""GraniteMoeShared model configuration"""
logger = ...

class GraniteMoeSharedConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
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
        embedding_multiplier=...,
        logits_scaling=...,
        residual_multiplier=...,
        attention_multiplier=...,
        num_local_experts=...,
        num_experts_per_tok=...,
        output_router_logits=...,
        router_aux_loss_coef=...,
        shared_intermediate_size=...,
        **kwargs,
    ) -> None: ...

__all__ = ["GraniteMoeSharedConfig"]
