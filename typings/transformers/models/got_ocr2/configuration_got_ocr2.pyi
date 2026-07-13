from ...configuration_utils import PretrainedConfig

class GotOcr2VisionConfig(PretrainedConfig):
    base_config_key = ...
    def __init__(
        self,
        hidden_size=...,
        output_channels=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_channels=...,
        image_size=...,
        patch_size=...,
        hidden_act=...,
        layer_norm_eps=...,
        attention_dropout=...,
        initializer_range=...,
        qkv_bias=...,
        use_abs_pos=...,
        use_rel_pos=...,
        window_size=...,
        global_attn_indexes=...,
        mlp_dim=...,
        **kwargs,
    ) -> None: ...

class GotOcr2Config(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    sub_configs = ...
    def __init__(
        self,
        vision_config=...,
        text_config=...,
        image_token_index=...,
        image_seq_length=...,
        pad_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["GotOcr2Config", "GotOcr2VisionConfig"]
