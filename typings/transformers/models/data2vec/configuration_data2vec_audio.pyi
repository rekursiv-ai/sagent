from ...configuration_utils import PretrainedConfig

"""Data2VecText configuration"""
logger = ...

class Data2VecAudioConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout=...,
        activation_dropout=...,
        attention_dropout=...,
        feat_proj_dropout=...,
        final_dropout=...,
        layerdrop=...,
        initializer_range=...,
        layer_norm_eps=...,
        feat_extract_activation=...,
        conv_dim=...,
        conv_stride=...,
        conv_kernel=...,
        conv_bias=...,
        num_conv_pos_embedding_groups=...,
        conv_pos_kernel_size=...,
        num_conv_pos_embeddings=...,
        mask_time_prob=...,
        mask_time_length=...,
        mask_time_min_masks=...,
        mask_feature_prob=...,
        mask_feature_length=...,
        mask_feature_min_masks=...,
        ctc_loss_reduction=...,
        ctc_zero_infinity=...,
        use_weighted_layer_sum=...,
        classifier_proj_size=...,
        tdnn_dim=...,
        tdnn_kernel=...,
        tdnn_dilation=...,
        xvector_output_dim=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        add_adapter=...,
        adapter_kernel_size=...,
        adapter_stride=...,
        num_adapter_layers=...,
        output_hidden_size=...,
        **kwargs,
    ) -> None: ...
    @property
    def inputs_to_logits_ratio(self):  # -> int:
        ...

__all__ = ["Data2VecAudioConfig"]
