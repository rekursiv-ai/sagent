import weakref

from torch import nn

class _State: ...

_module_state_mapping: weakref.WeakKeyDictionary[
    nn.Module, weakref.ReferenceType[_State]
] = ...
