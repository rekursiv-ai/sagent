from torch.optim import (
    lr_scheduler as lr_scheduler,
    swa_utils as swa_utils,
)
from torch.optim._adafactor import Adafactor as Adafactor
from torch.optim._muon import Muon as Muon
from torch.optim.adadelta import Adadelta as Adadelta
from torch.optim.adagrad import Adagrad as Adagrad
from torch.optim.adam import Adam as Adam
from torch.optim.adamax import Adamax as Adamax
from torch.optim.adamw import AdamW as AdamW
from torch.optim.asgd import ASGD as ASGD
from torch.optim.lbfgs import LBFGS as LBFGS
from torch.optim.nadam import NAdam as NAdam
from torch.optim.optimizer import Optimizer as Optimizer
from torch.optim.radam import RAdam as RAdam
from torch.optim.rmsprop import RMSprop as RMSprop
from torch.optim.rprop import Rprop as Rprop
from torch.optim.sgd import SGD as SGD
from torch.optim.sparse_adam import SparseAdam as SparseAdam

"""
:mod:`torch.optim` is a package implementing various optimization algorithms.

Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can also be easily integrated in the
future.
"""
__all__ = [
    "ASGD",
    "LBFGS",
    "SGD",
    "Adadelta",
    "Adafactor",
    "Adagrad",
    "Adam",
    "AdamW",
    "Adamax",
    "Muon",
    "NAdam",
    "Optimizer",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SparseAdam",
    "lr_scheduler",
    "swa_utils",
]
