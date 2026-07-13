from ...configuration_utils import PretrainedConfig

"""
TAPAS configuration. Based on the BERT configuration with added parameters.

Hyperparameters are taken from run_task_main.py and hparam_utils.py of the original implementation. URLS:

- https://github.com/google-research/tapas/blob/master/tapas/run_task_main.py
- https://github.com/google-research/tapas/blob/master/tapas/utils/hparam_utils.py

"""

class TapasConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        max_position_embeddings=...,
        type_vocab_sizes=...,
        initializer_range=...,
        layer_norm_eps=...,
        pad_token_id=...,
        positive_label_weight=...,
        num_aggregation_labels=...,
        aggregation_loss_weight=...,
        use_answer_as_supervision=...,
        answer_loss_importance=...,
        use_normalized_answer_loss=...,
        huber_loss_delta=...,
        temperature=...,
        aggregation_temperature=...,
        use_gumbel_for_cells=...,
        use_gumbel_for_aggregation=...,
        average_approximation_function=...,
        cell_selection_preference=...,
        answer_loss_cutoff=...,
        max_num_rows=...,
        max_num_columns=...,
        average_logits_per_cell=...,
        select_one_column=...,
        allow_empty_column_selection=...,
        init_cell_selection_weights_to_zero=...,
        reset_position_index_per_cell=...,
        disable_per_token_loss=...,
        aggregation_labels=...,
        no_aggregation_label_index=...,
        **kwargs,
    ) -> None: ...

__all__ = ["TapasConfig"]
