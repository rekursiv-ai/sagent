from typing import Any

import torch

__all__ = ["LSTM"]

class LSTM(torch.ao.nn.quantizable.LSTM):
    _FLOAT_MODULE = torch.ao.nn.quantizable.LSTM
    @classmethod
    def from_float(cls, *args: Any, **kwargs: Any) -> None: ...
    @classmethod
    def from_observed(cls: type[LSTM], other: torch.ao.nn.quantizable.LSTM) -> LSTM: ...
