from ...configuration_utils import PretrainedConfig

"""MRA model configuration"""
logger = ...

class MraConfig(PretrainedConfig):
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
        position_embedding_type=...,
        block_per_row=...,
        approx_mode=...,
        initial_prior_first_n_blocks=...,
        initial_prior_diagonal_n_blocks=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["MraConfig"]
