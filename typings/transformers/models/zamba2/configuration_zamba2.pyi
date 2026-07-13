from ...configuration_utils import PretrainedConfig

class Zamba2Config(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        max_position_embeddings=...,
        hidden_size=...,
        num_hidden_layers=...,
        layers_block_type=...,
        mamba_d_state=...,
        mamba_d_conv=...,
        mamba_expand=...,
        mamba_ngroups=...,
        time_step_min=...,
        time_step_max=...,
        time_step_floor=...,
        time_step_limit=...,
        n_mamba_heads=...,
        use_conv_bias=...,
        chunk_size=...,
        use_mem_eff_path=...,
        add_bias_linear=...,
        intermediate_size=...,
        hidden_act=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        attention_dropout=...,
        num_mem_blocks=...,
        use_shared_attention_adapter=...,
        adapter_rank=...,
        use_mem_rope=...,
        rope_theta=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        num_logits_to_keep=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        use_long_context=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Zamba2Config"]
