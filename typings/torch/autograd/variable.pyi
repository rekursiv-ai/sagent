import torch

__all__ = ["Variable", "VariableMeta"]

class VariableMeta(type):
    def __instancecheck__(cls, other) -> bool: ...

class Variable(torch._C._LegacyVariableBase, metaclass=VariableMeta):
    _execution_engine = ...
