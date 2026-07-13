from ...configuration_utils import PretrainedConfig

"""Switch Transformers model configuration"""
logger = ...

class SwitchTransformersConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        d_model=...,
        d_kv=...,
        d_ff=...,
        expert_capacity=...,
        num_layers=...,
        num_sparse_encoder_layers=...,
        num_decoder_layers=...,
        num_sparse_decoder_layers=...,
        num_heads=...,
        num_experts=...,
        router_bias=...,
        router_jitter_noise=...,
        router_dtype=...,
        router_ignore_padding_tokens=...,
        relative_attention_num_buckets=...,
        relative_attention_max_distance=...,
        dropout_rate=...,
        layer_norm_epsilon=...,
        router_z_loss_coef=...,
        router_aux_loss_coef=...,
        initializer_factor=...,
        dense_act_fn=...,
        is_encoder_decoder=...,
        add_router_probs=...,
        use_cache=...,
        pad_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["SwitchTransformersConfig"]
