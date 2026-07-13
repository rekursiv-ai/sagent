from ...configuration_utils import PretrainedConfig

"""PatchTSMixer model configuration"""
logger = ...

class PatchTSMixerConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        context_length: int = ...,
        patch_length: int = ...,
        num_input_channels: int = ...,
        patch_stride: int = ...,
        num_parallel_samples: int = ...,
        d_model: int = ...,
        expansion_factor: int = ...,
        num_layers: int = ...,
        dropout: float = ...,
        mode: str = ...,
        gated_attn: bool = ...,
        norm_mlp: str = ...,
        self_attn: bool = ...,
        self_attn_heads: int = ...,
        use_positional_encoding: bool = ...,
        positional_encoding_type: str = ...,
        scaling: str | bool | None = ...,
        loss: str = ...,
        init_std: float = ...,
        post_init: bool = ...,
        norm_eps: float = ...,
        mask_type: str = ...,
        random_mask_ratio: float = ...,
        num_forecast_mask_patches: list[int] | int | None = ...,
        mask_value: int = ...,
        masked_loss: bool = ...,
        channel_consistent_masking: bool = ...,
        unmasked_channel_indices: list[int] | None = ...,
        head_dropout: float = ...,
        distribution_output: str = ...,
        prediction_length: int = ...,
        prediction_channel_indices: list | None = ...,
        num_targets: int = ...,
        output_range: list | None = ...,
        head_aggregation: str = ...,
        **kwargs,
    ) -> None: ...

__all__ = ["PatchTSMixerConfig"]
