from transformers.configuration_utils import PretrainedConfig

"""PatchTST model configuration"""
logger = ...

class PatchTSTConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        num_input_channels: int = ...,
        context_length: int = ...,
        distribution_output: str = ...,
        loss: str = ...,
        patch_length: int = ...,
        patch_stride: int = ...,
        num_hidden_layers: int = ...,
        d_model: int = ...,
        num_attention_heads: int = ...,
        share_embedding: bool = ...,
        channel_attention: bool = ...,
        ffn_dim: int = ...,
        norm_type: str = ...,
        norm_eps: float = ...,
        attention_dropout: float = ...,
        positional_dropout: float = ...,
        path_dropout: float = ...,
        ff_dropout: float = ...,
        bias: bool = ...,
        activation_function: str = ...,
        pre_norm: bool = ...,
        positional_encoding_type: str = ...,
        use_cls_token: bool = ...,
        init_std: float = ...,
        share_projection: bool = ...,
        scaling: str | bool | None = ...,
        do_mask_input: bool | None = ...,
        mask_type: str = ...,
        random_mask_ratio: float = ...,
        num_forecast_mask_patches: list[int] | int | None = ...,
        channel_consistent_masking: bool | None = ...,
        unmasked_channel_indices: list[int] | None = ...,
        mask_value: int = ...,
        pooling_type: str = ...,
        head_dropout: float = ...,
        prediction_length: int = ...,
        num_targets: int = ...,
        output_range: list | None = ...,
        num_parallel_samples: int = ...,
        **kwargs,
    ) -> None: ...

__all__ = ["PatchTSTConfig"]
