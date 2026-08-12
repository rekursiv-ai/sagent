from ...configuration_utils import PretrainedConfig

class Qwen3VLMoeTextConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        tie_word_embeddings=...,
        rope_theta=...,
        attention_bias=...,
        attention_dropout=...,
        decoder_sparse_step=...,
        moe_intermediate_size=...,
        num_experts_per_tok=...,
        num_experts=...,
        norm_topk_prob=...,
        router_aux_loss_coef=...,
        mlp_only_layers=...,
        rope_scaling=...,
        head_dim=...,
        **kwargs,
    ) -> None: ...

class Qwen3VLMoeVisionConfig(PretrainedConfig):
    def __init__(
        self,
        depth=...,
        hidden_size=...,
        hidden_act=...,
        intermediate_size=...,
        num_heads=...,
        in_channels=...,
        patch_size=...,
        spatial_merge_size=...,
        temporal_patch_size=...,
        out_hidden_size=...,
        num_position_embeddings=...,
        deepstack_visual_indexes=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class Qwen3VLMoeConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        text_config=...,
        vision_config=...,
        image_token_id=...,
        video_token_id=...,
        vision_start_token_id=...,
        vision_end_token_id=...,
        tie_word_embeddings=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Qwen3VLMoeConfig", "Qwen3VLMoeTextConfig"]
