from ...configuration_utils import PretrainedConfig

class AriaTextConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    base_model_tp_plan = ...
    base_model_pp_plan = ...
    base_config_key = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size: int = ...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        pretraining_tp=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        attention_bias=...,
        attention_dropout=...,
        mlp_bias=...,
        head_dim=...,
        moe_num_experts: int = ...,
        moe_topk: int = ...,
        moe_num_shared_experts: int = ...,
        **kwargs,
    ) -> None: ...

class AriaConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    sub_configs = ...
    def __init__(
        self,
        vision_config=...,
        vision_feature_layer: int = ...,
        text_config: AriaTextConfig = ...,
        projector_patch_to_query_dict: dict | None = ...,
        image_token_index: int = ...,
        initializer_range: float = ...,
        **kwargs,
    ) -> None: ...

__all__ = ["AriaConfig", "AriaTextConfig"]
