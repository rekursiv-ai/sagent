from ...configuration_utils import PretrainedConfig

"""CLIPSeg model configuration"""
logger = ...

class CLIPSegTextConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        max_position_embeddings=...,
        hidden_act=...,
        layer_norm_eps=...,
        attention_dropout=...,
        initializer_range=...,
        initializer_factor=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

class CLIPSegVisionConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_channels=...,
        image_size=...,
        patch_size=...,
        hidden_act=...,
        layer_norm_eps=...,
        attention_dropout=...,
        initializer_range=...,
        initializer_factor=...,
        **kwargs,
    ) -> None: ...

class CLIPSegConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        text_config=...,
        vision_config=...,
        projection_dim=...,
        logit_scale_init_value=...,
        extract_layers=...,
        reduce_dim=...,
        decoder_num_attention_heads=...,
        decoder_attention_dropout=...,
        decoder_hidden_act=...,
        decoder_intermediate_size=...,
        conditional_layer=...,
        use_complex_transposed_convolution=...,
        **kwargs,
    ) -> None: ...

__all__ = ["CLIPSegConfig", "CLIPSegTextConfig", "CLIPSegVisionConfig"]
