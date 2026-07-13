from ...configuration_utils import PretrainedConfig

class EomtConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        mlp_ratio=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        initializer_range=...,
        layer_norm_eps=...,
        image_size=...,
        patch_size=...,
        num_channels=...,
        layerscale_value=...,
        drop_path_rate=...,
        num_upscale_blocks=...,
        attention_dropout=...,
        use_swiglu_ffn=...,
        num_blocks=...,
        no_object_weight: float = ...,
        class_weight: float = ...,
        mask_weight: float = ...,
        dice_weight: float = ...,
        train_num_points: int = ...,
        oversample_ratio: float = ...,
        importance_sample_ratio: float = ...,
        num_queries=...,
        num_register_tokens=...,
        **kwargs,
    ) -> None: ...

__all__ = ["EomtConfig"]
