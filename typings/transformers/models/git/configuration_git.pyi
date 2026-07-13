from ...configuration_utils import PretrainedConfig

logger = ...

class GitVisionConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_channels=...,
        image_size=...,
        patch_size=...,
        hidden_act=...,
        layer_norm_eps=...,
        attention_dropout=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class GitConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        vision_config=...,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        max_position_embeddings=...,
        initializer_range=...,
        layer_norm_eps=...,
        pad_token_id=...,
        position_embedding_type=...,
        use_cache=...,
        tie_word_embeddings=...,
        bos_token_id=...,
        eos_token_id=...,
        num_image_with_embedding=...,
        **kwargs,
    ) -> None: ...

__all__ = ["GitConfig", "GitVisionConfig"]
