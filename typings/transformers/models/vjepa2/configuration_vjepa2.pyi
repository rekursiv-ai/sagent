from ...configuration_utils import PretrainedConfig

"""VJEPA 2 model configuration"""

class VJEPA2Config(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        patch_size=...,
        crop_size=...,
        frames_per_clip=...,
        tubelet_size=...,
        hidden_size=...,
        in_chans=...,
        num_attention_heads=...,
        num_hidden_layers=...,
        drop_path_rate=...,
        mlp_ratio=...,
        layer_norm_eps=...,
        qkv_bias=...,
        attention_probs_dropout_prob=...,
        hidden_act=...,
        initializer_range=...,
        attention_dropout=...,
        num_pooler_layers=...,
        pred_hidden_size=...,
        pred_num_attention_heads=...,
        pred_num_hidden_layers=...,
        pred_num_mask_tokens=...,
        pred_zero_init_mask_tokens=...,
        pred_mlp_ratio=...,
        **kwargs,
    ) -> None: ...

__all__ = ["VJEPA2Config"]
