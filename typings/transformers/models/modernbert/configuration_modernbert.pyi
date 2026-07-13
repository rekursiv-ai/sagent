from typing import Literal

from ...configuration_utils import PretrainedConfig

class ModernBertConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        hidden_activation=...,
        max_position_embeddings=...,
        initializer_range=...,
        initializer_cutoff_factor=...,
        norm_eps=...,
        norm_bias=...,
        pad_token_id=...,
        eos_token_id=...,
        bos_token_id=...,
        cls_token_id=...,
        sep_token_id=...,
        global_rope_theta=...,
        attention_bias=...,
        attention_dropout=...,
        global_attn_every_n_layers=...,
        local_attention=...,
        local_rope_theta=...,
        embedding_dropout=...,
        mlp_bias=...,
        mlp_dropout=...,
        decoder_bias=...,
        classifier_pooling: Literal["cls", "mean"] = ...,
        classifier_dropout=...,
        classifier_bias=...,
        classifier_activation=...,
        deterministic_flash_attn=...,
        sparse_prediction=...,
        sparse_pred_ignore_index=...,
        reference_compile=...,
        repad_logits_with_grad=...,
        **kwargs,
    ) -> None: ...
    def to_dict(self):  # -> dict[str, Any]:
        ...

__all__ = ["ModernBertConfig"]
