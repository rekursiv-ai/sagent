from ....configuration_utils import PretrainedConfig

"""Graphormer model configuration"""
logger = ...

class GraphormerConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        num_classes: int = ...,
        num_atoms: int = ...,
        num_edges: int = ...,
        num_in_degree: int = ...,
        num_out_degree: int = ...,
        num_spatial: int = ...,
        num_edge_dis: int = ...,
        multi_hop_max_dist: int = ...,
        spatial_pos_max: int = ...,
        edge_type: str = ...,
        max_nodes: int = ...,
        share_input_output_embed: bool = ...,
        num_hidden_layers: int = ...,
        embedding_dim: int = ...,
        ffn_embedding_dim: int = ...,
        num_attention_heads: int = ...,
        dropout: float = ...,
        attention_dropout: float = ...,
        activation_dropout: float = ...,
        layerdrop: float = ...,
        encoder_normalize_before: bool = ...,
        pre_layernorm: bool = ...,
        apply_graphormer_init: bool = ...,
        activation_fn: str = ...,
        embed_scale: float | None = ...,
        freeze_embeddings: bool = ...,
        num_trans_layers_to_freeze: int = ...,
        traceable: bool = ...,
        q_noise: float = ...,
        qn_block_size: int = ...,
        kdim: int | None = ...,
        vdim: int | None = ...,
        bias: bool = ...,
        self_attention: bool = ...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["GraphormerConfig"]
