from ....configuration_utils import PretrainedConfig

"""REALM model configuration."""
logger = ...

class RealmConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        retriever_proj_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_candidates=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        max_position_embeddings=...,
        type_vocab_size=...,
        initializer_range=...,
        layer_norm_eps=...,
        span_hidden_size=...,
        max_span_width=...,
        reader_layer_norm_eps=...,
        reader_beam_size=...,
        reader_seq_len=...,
        num_block_records=...,
        searcher_beam_size=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["RealmConfig"]
