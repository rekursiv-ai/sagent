from ...configuration_utils import PretrainedConfig

"""Wav2Vec2Bert model configuration"""
logger = ...

class Wav2Vec2BertConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_size=...,
        feature_projection_input_dim=...,
        hidden_act=...,
        hidden_dropout=...,
        activation_dropout=...,
        attention_dropout=...,
        feat_proj_dropout=...,
        final_dropout=...,
        layerdrop=...,
        initializer_range=...,
        layer_norm_eps=...,
        apply_spec_augment=...,
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
        adapter_act=...,
        use_intermediate_ffn_before_adapter=...,
        output_hidden_size=...,
        position_embeddings_type=...,
        rotary_embedding_base=...,
        max_source_positions=...,
        left_max_position_embeddings=...,
        right_max_position_embeddings=...,
        conv_depthwise_kernel_size=...,
        conformer_conv_dropout=...,
        **kwargs,
    ) -> None: ...
    @property
    def inputs_to_logits_ratio(self):  # -> Any | int:
        ...

__all__ = ["Wav2Vec2BertConfig"]
