from torch import nn

import torch

class RTDetrHungarianMatcher(nn.Module):
    def __init__(self, config) -> None: ...
    @torch.no_grad()
    def forward(self, outputs, targets):  # -> list[tuple[Tensor, Tensor]]:
        ...

class RTDetrLoss(nn.Module):
    def __init__(self, config) -> None: ...
    def loss_labels_vfl(
        self, outputs, targets, indices, num_boxes, log=...
    ):  # -> dict[str, Any]:
        ...
    def loss_labels(
        self, outputs, targets, indices, num_boxes, log=...
    ):  # -> dict[str, Tensor]:
        ...
    @torch.no_grad()
    def loss_cardinality(
        self, outputs, targets, indices, num_boxes
    ):  # -> dict[str, Tensor]:
        ...
    def loss_boxes(self, outputs, targets, indices, num_boxes):  # -> dict[Any, Any]:
        ...
    def loss_masks(self, outputs, targets, indices, num_boxes):  # -> dict[str, Any]:
        ...
    def loss_labels_bce(
        self, outputs, targets, indices, num_boxes, log=...
    ):  # -> dict[str, Any]:
        ...
    def loss_labels_focal(
        self, outputs, targets, indices, num_boxes, log=...
    ):  # -> dict[str, Any]:
        ...
    def get_loss(
        self, loss, outputs, targets, indices, num_boxes
    ):  # -> dict[str, Tensor] | dict[str, Any] | dict[Any, Any]:
        ...
    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):  # -> list[Any]:
        ...
    def forward(self, outputs, targets):  # -> dict[Any, Any]:
        ...

def RTDetrForObjectDetectionLoss(
    logits,
    labels,
    device,
    pred_boxes,
    config,
    outputs_class=...,
    outputs_coord=...,
    enc_topk_logits=...,
    enc_topk_bboxes=...,
    denoising_meta_values=...,
    **kwargs,
):  # -> tuple[int, Any, list[dict[str, Any]] | Any]:
    ...
