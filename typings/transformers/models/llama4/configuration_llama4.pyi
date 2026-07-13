from ...configuration_utils import PretrainedConfig

logger = ...

class Llama4VisionConfig(PretrainedConfig):
    base_model_tp_plan = ...
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        hidden_size: int = ...,
        hidden_act: str = ...,
        num_hidden_layers: int = ...,
        num_attention_heads: int = ...,
        num_channels: int = ...,
        intermediate_size: int = ...,
        vision_output_dim: int = ...,
        image_size: int = ...,
        patch_size: int = ...,
        norm_eps: float = ...,
        vision_feature_select_strategy=...,
        initializer_range: float = ...,
        pixel_shuffle_ratio=...,
        projector_input_dim=...,
        projector_output_dim=...,
        multi_modal_projector_bias=...,
        projector_dropout=...,
        attention_dropout=...,
        rope_theta=...,
        **kwargs,
    ) -> None: ...

class Llama4TextConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    base_model_tp_plan = ...
    base_model_ep_plan = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        intermediate_size_mlp=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        head_dim=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        tie_word_embeddings=...,
        rope_theta=...,
        attention_dropout=...,
        num_experts_per_tok=...,
        num_local_experts=...,
        moe_layers=...,
        interleave_moe_layer_step=...,
        use_qk_norm=...,
        output_router_logits=...,
        router_aux_loss_coef=...,
        router_jitter_noise=...,
        rope_scaling=...,
        no_rope_layers=...,
        no_rope_layer_interval=...,
        attention_chunk_size=...,
        layer_types=...,
        attn_temperature_tuning=...,
        floor_scale=...,
        attn_scale=...,
        **kwargs,
    ) -> None: ...

class Llama4Config(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    sub_configs = ...
    base_model_tp_plan = ...
    def __init__(
        self,
        vision_config=...,
        text_config=...,
        boi_token_index=...,
        eoi_token_index=...,
        image_token_index=...,
        tie_word_embeddings=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Llama4Config", "Llama4TextConfig", "Llama4VisionConfig"]
