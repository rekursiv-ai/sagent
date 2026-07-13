from ...configuration_utils import PretrainedConfig

"""Pop2Piano model configuration"""
logger = ...

class Pop2PianoConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        composer_vocab_size=...,
        d_model=...,
        d_kv=...,
        d_ff=...,
        num_layers=...,
        num_decoder_layers=...,
        num_heads=...,
        relative_attention_num_buckets=...,
        relative_attention_max_distance=...,
        dropout_rate=...,
        layer_norm_epsilon=...,
        initializer_factor=...,
        feed_forward_proj=...,
        is_encoder_decoder=...,
        use_cache=...,
        pad_token_id=...,
        eos_token_id=...,
        dense_act_fn=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Pop2PianoConfig"]
