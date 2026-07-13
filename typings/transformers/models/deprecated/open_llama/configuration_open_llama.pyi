from ....configuration_utils import PretrainedConfig

"""Open-Llama model configuration"""
logger = ...

class OpenLlamaConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        tie_word_embeddings=...,
        use_memory_efficient_attention=...,
        hidden_dropout_prob=...,
        attention_dropout_prob=...,
        use_stable_embedding=...,
        shared_input_output_embedding=...,
        rope_theta=...,
        rope_scaling=...,
        **kwargs,
    ) -> None: ...

__all__ = ["OpenLlamaConfig"]
