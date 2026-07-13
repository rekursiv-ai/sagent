import torch

from .loss_for_object_detection import HungarianMatcher, ImageLoss

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: int,
    alpha: float = ...,
    gamma: float = ...,
):  # -> Tensor:
    ...

class GroundingDinoHungarianMatcher(HungarianMatcher):
    @torch.no_grad()
    def forward(self, outputs, targets):  # -> list[tuple[Tensor, Tensor]]:
        ...

class GroundingDinoImageLoss(ImageLoss):
    def __init__(self, matcher, focal_alpha, losses) -> None: ...
    def loss_labels(
        self, outputs, targets, indices, num_boxes
    ):  # -> dict[str, Tensor]:
        ...

def GroundingDinoForObjectDetectionLoss(
    logits,
    labels,
    device,
    pred_boxes,
    config,
    label_maps,
    text_mask,
    outputs_class=...,
    outputs_coord=...,
    encoder_logits=...,
    encoder_pred_boxes=...,
):  # -> tuple[int, Any, list[dict[str, Any]] | None]:
    ...
