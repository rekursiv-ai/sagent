from ...configuration_utils import PretrainedConfig

class EfficientLoFTRConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        stage_num_blocks: list[int] | None = ...,
        out_features: list[int] | None = ...,
        stage_stride: list[int] | None = ...,
        hidden_size: int = ...,
        activation_function: str = ...,
        q_aggregation_kernel_size: int = ...,
        kv_aggregation_kernel_size: int = ...,
        q_aggregation_stride: int = ...,
        kv_aggregation_stride: int = ...,
        num_attention_layers: int = ...,
        num_attention_heads: int = ...,
        attention_dropout: float = ...,
        attention_bias: bool = ...,
        mlp_activation_function: str = ...,
        coarse_matching_skip_softmax: bool = ...,
        coarse_matching_threshold: float = ...,
        coarse_matching_temperature: float = ...,
        coarse_matching_border_removal: int = ...,
        fine_kernel_size: int = ...,
        batch_norm_eps: float = ...,
        rope_theta: float = ...,
        partial_rotary_factor: float = ...,
        rope_scaling: dict | None = ...,
        fine_matching_slice_dim: int = ...,
        fine_matching_regress_temperature: float = ...,
        initializer_range: float = ...,
        **kwargs,
    ) -> None: ...

__all__ = ["EfficientLoFTRConfig"]
