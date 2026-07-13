import torch

def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: torch.Tensor | None = ...,
    ignore_index: int = ...,
    **kwargs,
) -> torch.Tensor: ...
def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: torch.Tensor | None = ...,
    ignore_index: int = ...,
    shift_labels: torch.Tensor | None = ...,
    **kwargs,
) -> torch.Tensor: ...
def ForMaskedLMLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: torch.Tensor | None = ...,
    ignore_index: int = ...,
    **kwargs,
):  # -> Tensor:
    ...
def ForSequenceClassificationLoss(
    labels: torch.Tensor, pooled_logits: torch.Tensor, config, **kwargs
) -> torch.Tensor: ...
def ForQuestionAnsweringLoss(
    start_logits, end_logits, start_positions, end_positions, **kwargs
):  # -> Tensor | None:
    ...
def ForTokenClassification(
    logits: torch.Tensor, labels, config, **kwargs
):  # -> Tensor:
    ...

LOSS_MAPPING = ...
