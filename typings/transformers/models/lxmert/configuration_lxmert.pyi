from ...configuration_utils import PretrainedConfig

"""LXMERT model configuration"""
logger = ...

class LxmertConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_attention_heads=...,
        num_qa_labels=...,
        num_object_labels=...,
        num_attr_labels=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        max_position_embeddings=...,
        type_vocab_size=...,
        initializer_range=...,
        layer_norm_eps=...,
        l_layers=...,
        x_layers=...,
        r_layers=...,
        visual_feat_dim=...,
        visual_pos_dim=...,
        visual_loss_normalizer=...,
        task_matched=...,
        task_mask_lm=...,
        task_obj_predict=...,
        task_qa=...,
        visual_obj_loss=...,
        visual_attr_loss=...,
        visual_feat_loss=...,
        **kwargs,
    ) -> None: ...

__all__ = ["LxmertConfig"]
