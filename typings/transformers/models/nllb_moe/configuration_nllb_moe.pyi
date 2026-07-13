from ...configuration_utils import PretrainedConfig

"""NLLB-MoE model configuration"""
logger = ...

class NllbMoeConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        max_position_embeddings=...,
        encoder_layers=...,
        encoder_ffn_dim=...,
        encoder_attention_heads=...,
        decoder_layers=...,
        decoder_ffn_dim=...,
        decoder_attention_heads=...,
        encoder_layerdrop=...,
        decoder_layerdrop=...,
        use_cache=...,
        is_encoder_decoder=...,
        activation_function=...,
        d_model=...,
        dropout=...,
        attention_dropout=...,
        activation_dropout=...,
        init_std=...,
        decoder_start_token_id=...,
        scale_embedding=...,
        router_bias=...,
        router_dtype=...,
        router_ignore_padding_tokens=...,
        num_experts=...,
        expert_capacity=...,
        encoder_sparse_step=...,
        decoder_sparse_step=...,
        router_z_loss_coef=...,
        router_aux_loss_coef=...,
        second_expert_policy=...,
        normalize_router_prob_before_dropping=...,
        batch_prioritized_routing=...,
        moe_eval_capacity_token_fraction=...,
        moe_token_dropout=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        output_router_logits=...,
        **kwargs,
    ) -> None: ...

__all__ = ["NllbMoeConfig"]
