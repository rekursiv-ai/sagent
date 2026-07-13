from ...configuration_utils import PretrainedConfig

"""Nystromformer model configuration"""
logger = ...

class NystromformerConfig(PretrainedConfig):
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
        segment_means_seq_len=...,
        num_landmarks=...,
        conv_kernel_size=...,
        inv_coeff_init_option=...,
        initializer_range=...,
        layer_norm_eps=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["NystromformerConfig"]
