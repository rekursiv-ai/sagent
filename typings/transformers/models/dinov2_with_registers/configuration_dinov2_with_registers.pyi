from ...configuration_utils import PretrainedConfig
from ...utils.backbone_utils import BackboneConfigMixin

class Dinov2WithRegistersConfig(BackboneConfigMixin, PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        mlp_ratio=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        initializer_range=...,
        layer_norm_eps=...,
        image_size=...,
        patch_size=...,
        num_channels=...,
        qkv_bias=...,
        layerscale_value=...,
        drop_path_rate=...,
        use_swiglu_ffn=...,
        num_register_tokens=...,
        out_features=...,
        out_indices=...,
        apply_layernorm=...,
        reshape_hidden_states=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Dinov2WithRegistersConfig"]
