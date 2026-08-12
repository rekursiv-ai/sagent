from ...configuration_utils import PretrainedConfig

"""Evolla model configuration"""
logger = ...

class SaProtConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=...,
        mask_token_id=...,
        pad_token_id=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_size=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        max_position_embeddings=...,
        initializer_range=...,
        layer_norm_eps=...,
        position_embedding_type=...,
        emb_layer_norm_before=...,
        token_dropout=...,
        **kwargs,
    ) -> None: ...

class EvollaConfig(PretrainedConfig):
    def __init__(
        self,
        protein_encoder_config=...,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        hidden_act=...,
        max_position_embeddings=...,
        rms_norm_eps=...,
        rope_theta=...,
        rope_scaling=...,
        attention_bias=...,
        attention_dropout=...,
        mlp_bias=...,
        aligner_ffn_mult=...,
        aligner_enable_bias=...,
        aligner_attention_probs_dropout_prob=...,
        aligner_num_add_layers=...,
        resampler_depth=...,
        resampler_dim_head=...,
        resampler_heads=...,
        resampler_num_latents=...,
        resampler_ff_mult=...,
        initializer_range=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        use_cache=...,
        tie_word_embeddings=...,
        **kwargs,
    ) -> None: ...

__all__ = ["EvollaConfig"]
