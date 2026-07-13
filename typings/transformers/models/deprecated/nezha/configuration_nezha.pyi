from .... import PretrainedConfig

class NezhaConfig(PretrainedConfig):
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
        max_relative_position=...,
        type_vocab_size=...,
        initializer_range=...,
        layer_norm_eps=...,
        classifier_dropout=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        use_cache=...,
        **kwargs,
    ) -> None: ...

__all__ = ["NezhaConfig"]
