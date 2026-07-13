from ...configuration_utils import PretrainedConfig

"""RoCBert model configuration"""
logger = ...

class RoCBertConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        max_position_embeddings=...,
        type_vocab_size=...,
        initializer_range=...,
        layer_norm_eps=...,
        use_cache=...,
        pad_token_id=...,
        position_embedding_type=...,
        classifier_dropout=...,
        enable_pronunciation=...,
        enable_shape=...,
        pronunciation_embed_dim=...,
        pronunciation_vocab_size=...,
        shape_embed_dim=...,
        shape_vocab_size=...,
        concat_input=...,
        **kwargs,
    ) -> None: ...

__all__ = ["RoCBertConfig"]
