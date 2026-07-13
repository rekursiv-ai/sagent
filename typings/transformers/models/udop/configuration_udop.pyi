from ...configuration_utils import PretrainedConfig

"""UDOP model configuration"""
logger = ...

class UdopConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        d_model=...,
        d_kv=...,
        d_ff=...,
        num_layers=...,
        num_decoder_layers=...,
        num_heads=...,
        relative_attention_num_buckets=...,
        relative_attention_max_distance=...,
        relative_bias_args=...,
        dropout_rate=...,
        layer_norm_epsilon=...,
        initializer_factor=...,
        feed_forward_proj=...,
        is_encoder_decoder=...,
        use_cache=...,
        pad_token_id=...,
        eos_token_id=...,
        max_2d_position_embeddings=...,
        image_size=...,
        patch_size=...,
        num_channels=...,
        **kwargs,
    ) -> None: ...

__all__ = ["UdopConfig"]
