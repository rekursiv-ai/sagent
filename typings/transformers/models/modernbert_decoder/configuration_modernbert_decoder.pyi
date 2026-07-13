from ...configuration_utils import PretrainedConfig

class ModernBertDecoderConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        hidden_activation=...,
        max_position_embeddings=...,
        initializer_range=...,
        initializer_cutoff_factor=...,
        norm_eps=...,
        norm_bias=...,
        pad_token_id=...,
        eos_token_id=...,
        bos_token_id=...,
        cls_token_id=...,
        sep_token_id=...,
        global_rope_theta=...,
        attention_bias=...,
        attention_dropout=...,
        embedding_dropout=...,
        mlp_bias=...,
        mlp_dropout=...,
        decoder_bias=...,
        classifier_dropout=...,
        classifier_bias=...,
        classifier_activation=...,
        use_cache=...,
        local_attention=...,
        global_attn_every_n_layers=...,
        local_rope_theta=...,
        layer_types=...,
        **kwargs,
    ) -> None: ...

__all__ = ["ModernBertDecoderConfig"]
