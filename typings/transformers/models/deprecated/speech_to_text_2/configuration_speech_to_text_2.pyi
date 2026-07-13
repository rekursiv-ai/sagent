from ....configuration_utils import PretrainedConfig

"""Speech2Text model configuration"""
logger = ...

class Speech2Text2Config(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        decoder_layers=...,
        decoder_ffn_dim=...,
        decoder_attention_heads=...,
        decoder_layerdrop=...,
        use_cache=...,
        activation_function=...,
        d_model=...,
        dropout=...,
        attention_dropout=...,
        activation_dropout=...,
        init_std=...,
        decoder_start_token_id=...,
        scale_embedding=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        max_target_positions=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Speech2Text2Config"]
