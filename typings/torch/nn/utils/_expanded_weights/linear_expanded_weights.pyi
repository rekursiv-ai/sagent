from torch import Tensor
import torch
import torch.nn.functional as F

from .expanded_weights_impl import implements_per_sample_grads

@implements_per_sample_grads(F.linear)
class LinearPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _, __, *expanded_args_and_kwargs): ...
    @staticmethod
    def backward(ctx, grad_output) -> tuple[Tensor | None, ...]: ...
