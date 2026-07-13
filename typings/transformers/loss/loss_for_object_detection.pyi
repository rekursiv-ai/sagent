from torch import Tensor, nn

import torch

def dice_loss(inputs, targets, num_boxes): ...
def sigmoid_focal_loss(
    inputs, targets, num_boxes, alpha: float = ..., gamma: float = ...
): ...

class ImageLoss(nn.Module):
    def __init__(self, matcher, num_classes, eos_coef, losses) -> None: ...
    def loss_labels(
        self, outputs, targets, indices, num_boxes
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
    def get_loss(
        self, loss, outputs, targets, indices, num_boxes
    ):  # -> dict[str, Tensor] | dict[str, Any] | dict[Any, Any]:
        ...
    def forward(self, outputs, targets):  # -> dict[Any, Any]:
        ...

class HungarianMatcher(nn.Module):
    def __init__(
        self, class_cost: float = ..., bbox_cost: float = ..., giou_cost: float = ...
    ) -> None: ...
    @torch.no_grad()
    def forward(self, outputs, targets):  # -> list[tuple[Tensor, Tensor]]:
        ...

def box_area(boxes: Tensor) -> Tensor: ...
def box_iou(boxes1, boxes2):  # -> tuple[Any, Any]:
    ...
def generalized_box_iou(boxes1, boxes2): ...

class NestedTensor:
    def __init__(self, tensors, mask: Tensor | None) -> None: ...
    def to(self, device):  # -> NestedTensor:
        ...
    def decompose(self):  # -> tuple[Any, Tensor | None]:
        ...
    def __repr__(self):  # -> str:
        ...

def nested_tensor_from_tensor_list(tensor_list: list[Tensor]):  # -> NestedTensor:
    ...
def ForSegmentationLoss(
    logits,
    labels,
    device,
    pred_boxes,
    pred_masks,
    config,
    outputs_class=...,
    outputs_coord=...,
    **kwargs,
):  # -> tuple[int, Any, list[dict[str, Any]] | None]:
    ...
def ForObjectDetectionLoss(
    logits,
    labels,
    device,
    pred_boxes,
    config,
    outputs_class=...,
    outputs_coord=...,
    **kwargs,
):  # -> tuple[int, Any, list[dict[str, Any]] | None]:
    ...
