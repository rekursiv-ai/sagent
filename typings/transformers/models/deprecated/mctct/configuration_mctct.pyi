from ....configuration_utils import PretrainedConfig

"""M-CTC-T model configuration"""
logger = ...

class MCTCTConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        intermediate_size=...,
        num_attention_heads=...,
        attention_head_dim=...,
        max_position_embeddings=...,
        layer_norm_eps=...,
        layerdrop=...,
        hidden_act=...,
        initializer_range=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        conv_glu_dim=...,
        conv_dropout=...,
        num_conv_layers=...,
        conv_kernel=...,
        conv_stride=...,
        input_feat_per_channel=...,
        input_channels=...,
        conv_channels=...,
        ctc_loss_reduction=...,
        ctc_zero_infinity=...,
        **kwargs,
    ) -> None: ...

__all__ = ["MCTCTConfig"]
