from ...configuration_utils import PretrainedConfig

"""OLMoE model configuration"""

class OlmoeConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        attention_bias=...,
        attention_dropout=...,
        clip_qkv=...,
        num_experts_per_tok=...,
        num_experts=...,
        output_router_logits=...,
        router_aux_loss_coef=...,
        norm_topk_prob=...,
        **kwargs,
    ) -> None: ...

__all__ = ["OlmoeConfig"]
