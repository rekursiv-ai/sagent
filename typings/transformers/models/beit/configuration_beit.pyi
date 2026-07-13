from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils.backbone_utils import BackboneConfigMixin

"""BEiT model configuration"""

class BeitConfig(BackboneConfigMixin, PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        initializer_range=...,
        layer_norm_eps=...,
        image_size=...,
        patch_size=...,
        num_channels=...,
        use_mask_token=...,
        use_absolute_position_embeddings=...,
        use_relative_position_bias=...,
        use_shared_relative_position_bias=...,
        layer_scale_init_value=...,
        drop_path_rate=...,
        use_mean_pooling=...,
        pool_scales=...,
        use_auxiliary_head=...,
        auxiliary_loss_weight=...,
        auxiliary_channels=...,
        auxiliary_num_convs=...,
        auxiliary_concat_input=...,
        semantic_loss_ignore_index=...,
        out_features=...,
        out_indices=...,
        add_fpn=...,
        reshape_hidden_states=...,
        **kwargs,
    ) -> None: ...

class BeitOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...

__all__ = ["BeitConfig", "BeitOnnxConfig"]
