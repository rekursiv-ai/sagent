from ...configuration_utils import PretrainedConfig

logger = ...

class KyutaiSpeechToTextConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    sub_configs = ...
    def __init__(
        self,
        codebook_vocab_size=...,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        max_position_embeddings=...,
        rope_theta=...,
        hidden_act=...,
        head_dim=...,
        initializer_range=...,
        use_cache=...,
        sliding_window=...,
        attention_dropout=...,
        ffn_dim=...,
        rms_norm_eps=...,
        num_codebooks=...,
        audio_bos_token_id=...,
        audio_pad_token_id=...,
        tie_word_embeddings=...,
        pad_token_id=...,
        bos_token_id=...,
        codec_config=...,
        **kwargs,
    ) -> None: ...

__all__ = ["KyutaiSpeechToTextConfig"]
