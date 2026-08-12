from ...configuration_utils import PretrainedConfig

"""FSMT configuration"""
logger = ...

class DecoderConfig(PretrainedConfig):
    def __init__(
        self, vocab_size=..., bos_token_id=..., is_encoder_decoder=..., **kwargs
    ) -> None: ...

class FSMTConfig(PretrainedConfig):
    def __init__(
        self,
        langs=...,
        src_vocab_size=...,
        tgt_vocab_size=...,
        activation_function=...,
        d_model=...,
        max_length=...,
        max_position_embeddings=...,
        encoder_ffn_dim=...,
        encoder_layers=...,
        encoder_attention_heads=...,
        encoder_layerdrop=...,
        decoder_ffn_dim=...,
        decoder_layers=...,
        decoder_attention_heads=...,
        decoder_layerdrop=...,
        attention_dropout=...,
        dropout=...,
        activation_dropout=...,
        init_std=...,
        decoder_start_token_id=...,
        is_encoder_decoder=...,
        scale_embedding=...,
        tie_word_embeddings=...,
        num_beams=...,
        length_penalty=...,
        early_stopping=...,
        use_cache=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        forced_eos_token_id=...,
        **common_kwargs,
    ) -> None: ...

__all__ = ["FSMTConfig"]
