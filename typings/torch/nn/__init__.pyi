from torch.nn import (
    attention as attention,
    functional as functional,
    init as init,
    modules as modules,
    parallel as parallel,
    parameter as parameter,
    utils as utils,
)
from torch.nn.modules import *
from torch.nn.parallel import DataParallel as DataParallel
from torch.nn.parameter import (
    Buffer as Buffer,
    Parameter as Parameter,
    UninitializedBuffer as UninitializedBuffer,
    UninitializedParameter as UninitializedParameter,
)

def factory_kwargs(kwargs) -> dict[Any, Any]: ...
