from ...configuration_utils import PretrainedConfig

"""Reformer model configuration"""
logger = ...

class ReformerConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        attention_head_size=...,
        attn_layers=...,
        axial_norm_std=...,
        axial_pos_embds=...,
        axial_pos_shape=...,
        axial_pos_embds_dim=...,
        chunk_size_lm_head=...,
        eos_token_id=...,
        feed_forward_size=...,
        hash_seed=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        hidden_size=...,
        initializer_range=...,
        is_decoder=...,
        layer_norm_eps=...,
        local_num_chunks_before=...,
        local_num_chunks_after=...,
        local_attention_probs_dropout_prob=...,
        local_attn_chunk_length=...,
        lsh_attn_chunk_length=...,
        lsh_attention_probs_dropout_prob=...,
        lsh_num_chunks_before=...,
        lsh_num_chunks_after=...,
        max_position_embeddings=...,
        num_attention_heads=...,
        num_buckets=...,
        num_hashes=...,
        pad_token_id=...,
        vocab_size=...,
        tie_word_embeddings=...,
        use_cache=...,
        classifier_dropout=...,
        **kwargs,
    ) -> None: ...

__all__ = ["ReformerConfig"]
