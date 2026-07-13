import torch

from .loss_rt_detr import RTDetrLoss

def weighting_function(
    max_num_bins: int, up: torch.Tensor, reg_scale: int
) -> torch.Tensor: ...
def translate_gt(
    gt: torch.Tensor, max_num_bins: int, reg_scale: int, up: torch.Tensor
):  # -> tuple[Tensor, Tensor, Tensor]:
    ...
def bbox2distance(
    points, bbox, max_num_bins, reg_scale, up, eps=...
):  # -> tuple[Tensor, Tensor, Tensor]:
    ...

class DFineLoss(RTDetrLoss):
    def __init__(self, config) -> None: ...
    def unimodal_distribution_focal_loss(
        self,
        pred,
        label,
        weight_right,
        weight_left,
        weight=...,
        reduction=...,
        avg_factor=...,
    ): ...
    def loss_local(
        self, outputs, targets, indices, num_boxes, T=...
    ):  # -> dict[Any, Any]:
        ...
    def get_loss(
        self, loss, outputs, targets, indices, num_boxes
    ):  # -> dict[str, Any] | dict[Any, Any] | dict[str, Tensor]:
        ...

def DFineForObjectDetectionLoss(
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
    predicted_corners=...,
    initial_reference_points=...,
    **kwargs,
):  # -> tuple[int, Any, list[dict[str, Any | None]] | None]:
    ...
