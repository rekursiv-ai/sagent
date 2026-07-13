from ...configuration_utils import PretrainedConfig

"""Pix2Struct model configuration"""
logger = ...

class Pix2StructTextConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        d_kv=...,
        d_ff=...,
        num_layers=...,
        num_heads=...,
        relative_attention_num_buckets=...,
        relative_attention_max_distance=...,
        dropout_rate=...,
        layer_norm_epsilon=...,
        initializer_factor=...,
        dense_act_fn=...,
        decoder_start_token_id=...,
        use_cache=...,
        pad_token_id=...,
        eos_token_id=...,
        tie_word_embeddings=...,
        is_decoder=...,
        **kwargs,
    ) -> None: ...

class Pix2StructVisionConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        hidden_size=...,
        patch_embed_hidden_size=...,
        d_ff=...,
        d_kv=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        dense_act_fn=...,
        layer_norm_eps=...,
        dropout_rate=...,
        attention_dropout=...,
        initializer_range=...,
        initializer_factor=...,
        seq_len=...,
        relative_attention_num_buckets=...,
        relative_attention_max_distance=...,
        **kwargs,
    ) -> None: ...

class Pix2StructConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        text_config=...,
        vision_config=...,
        initializer_factor=...,
        initializer_range=...,
        is_vqa=...,
        tie_word_embeddings=...,
        is_encoder_decoder=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Pix2StructConfig", "Pix2StructTextConfig", "Pix2StructVisionConfig"]
