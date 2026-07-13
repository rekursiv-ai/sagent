from ...configuration_utils import PretrainedConfig

"""FNet model configuration"""
logger = ...

class FNetConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        max_position_embeddings=...,
        type_vocab_size=...,
        initializer_range=...,
        layer_norm_eps=...,
        use_tpu_fourier_optimizations=...,
        tpu_short_seq_length=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["FNetConfig"]
