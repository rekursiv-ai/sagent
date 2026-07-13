from ...configuration_utils import PretrainedConfig

"""Blt model configuration"""
logger = ...

class BltLocalEncoderConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        cross_attn_all_layers=...,
        cross_attn_k=...,
        hidden_size_global=...,
        hidden_size=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        num_hidden_layers=...,
        rms_norm_eps=...,
        dropout=...,
        max_position_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        hidden_act=...,
        intermediate_size=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class BltLocalDecoderConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        cross_attn_all_layers=...,
        cross_attn_k=...,
        hidden_size_global=...,
        hidden_size=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        num_hidden_layers=...,
        rms_norm_eps=...,
        dropout=...,
        max_position_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        hidden_act=...,
        intermediate_size=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class BltGlobalTransformerConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        hidden_size=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        num_hidden_layers=...,
        rms_norm_eps=...,
        dropout=...,
        max_position_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        hidden_act=...,
        intermediate_size=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class BltPatcherConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        max_position_embeddings=...,
        rms_norm_eps=...,
        dropout=...,
        rope_theta=...,
        intermediate_size=...,
        rope_scaling=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class BltConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    sub_configs = ...
    def __init__(
        self,
        vocab_size=...,
        max_position_embeddings=...,
        patch_in_forward=...,
        patch_size=...,
        patching_mode=...,
        patching_threshold=...,
        patching_batch_size=...,
        max_patch_length=...,
        cross_attn_k=...,
        encoder_hash_byte_group_size=...,
        encoder_hash_byte_group_vocab=...,
        encoder_hash_byte_group_nb_functions=...,
        patcher_config=...,
        encoder_config=...,
        decoder_config=...,
        global_config=...,
        tie_word_embeddings=...,
        initializer_range=...,
        rope_theta=...,
        rope_scaling=...,
        **kwargs,
    ) -> None: ...

__all__ = [
    "BltConfig",
    "BltGlobalTransformerConfig",
    "BltLocalDecoderConfig",
    "BltLocalEncoderConfig",
    "BltPatcherConfig",
]
