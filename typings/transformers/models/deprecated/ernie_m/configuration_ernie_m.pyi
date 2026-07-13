from ....configuration_utils import PretrainedConfig

"""ErnieM model configuration"""

class ErnieMConfig(PretrainedConfig):
    model_type = ...
    attribute_map: dict[str, str] = ...
    def __init__(
        self,
        vocab_size: int = ...,
        hidden_size: int = ...,
        num_hidden_layers: int = ...,
        num_attention_heads: int = ...,
        intermediate_size: int = ...,
        hidden_act: str = ...,
        hidden_dropout_prob: float = ...,
        attention_probs_dropout_prob: float = ...,
        max_position_embeddings: int = ...,
        initializer_range: float = ...,
        pad_token_id: int = ...,
        layer_norm_eps: float = ...,
        classifier_dropout=...,
        act_dropout=...,
        **kwargs,
    ) -> None: ...

__all__ = ["ErnieMConfig"]
