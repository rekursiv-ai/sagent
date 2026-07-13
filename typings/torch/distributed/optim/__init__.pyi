from torch import optim as optim

from .apply_optimizer_in_backward import (
    _apply_optimizer_in_backward as _apply_optimizer_in_backward,
    _get_in_backward_optimizers as _get_in_backward_optimizers,
)
from .functional_adadelta import _FunctionalAdadelta as _FunctionalAdadelta
from .functional_adagrad import _FunctionalAdagrad as _FunctionalAdagrad
from .functional_adam import _FunctionalAdam as _FunctionalAdam
from .functional_adamax import _FunctionalAdamax as _FunctionalAdamax
from .functional_adamw import _FunctionalAdamW as _FunctionalAdamW
from .functional_rmsprop import _FunctionalRMSprop as _FunctionalRMSprop
from .functional_rprop import _FunctionalRprop as _FunctionalRprop
from .functional_sgd import _FunctionalSGD as _FunctionalSGD
from .named_optimizer import _NamedOptimizer as _NamedOptimizer
from .optimizer import DistributedOptimizer as DistributedOptimizer
from .post_localSGD_optimizer import PostLocalSGDOptimizer as PostLocalSGDOptimizer
from .utils import as_functional_optim as as_functional_optim
from .zero_redundancy_optimizer import (
    ZeroRedundancyOptimizer as ZeroRedundancyOptimizer,
)

"""
:mod:`torch.distributed.optim` exposes DistributedOptimizer, which takes a list
of remote parameters (:class:`~torch.distributed.rpc.RRef`) and runs the
optimizer locally on the workers where the parameters live.  The distributed
optimizer can use any of the local optimizer :ref:`optimizer-algorithms` to
apply the gradients on each worker.
"""
__all__ = [
    "DistributedOptimizer",
    "PostLocalSGDOptimizer",
    "ZeroRedundancyOptimizer",
    "as_functional_optim",
]
