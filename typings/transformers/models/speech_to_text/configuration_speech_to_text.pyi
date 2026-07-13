from ...configuration_utils import PretrainedConfig

"""Speech2Text model configuration"""
logger = ...

class Speech2TextConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        encoder_layers=...,
        encoder_ffn_dim=...,
        encoder_attention_heads=...,
        decoder_layers=...,
        decoder_ffn_dim=...,
        decoder_attention_heads=...,
        encoder_layerdrop=...,
        decoder_layerdrop=...,
        use_cache=...,
        is_encoder_decoder=...,
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
        max_source_positions=...,
        max_target_positions=...,
        num_conv_layers=...,
        conv_kernel_sizes=...,
        conv_channels=...,
        input_feat_per_channel=...,
        input_channels=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Speech2TextConfig"]
