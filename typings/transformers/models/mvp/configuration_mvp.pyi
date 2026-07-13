from ...configuration_utils import PretrainedConfig

"""MVP model configuration"""
logger = ...

class MvpConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        max_position_embeddings=...,
        encoder_layers=...,
        encoder_ffn_dim=...,
        encoder_attention_heads=...,
        decoder_layers=...,
        decoder_ffn_dim=...,
        decoder_attention_heads=...,
        encoder_layerdrop=...,
        decoder_layerdrop=...,
        activation_function=...,
        d_model=...,
        dropout=...,
        attention_dropout=...,
        activation_dropout=...,
        init_std=...,
        classifier_dropout=...,
        scale_embedding=...,
        use_cache=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        is_encoder_decoder=...,
        decoder_start_token_id=...,
        forced_eos_token_id=...,
        use_prompt=...,
        prompt_length=...,
        prompt_mid_dim=...,
        **kwargs,
    ) -> None: ...

__all__ = ["MvpConfig"]
