from torch._utils import ExceptionWrapper as ExceptionWrapper

from . import (
    collate as collate,
    fetch as fetch,
    pin_memory as pin_memory,
    signal_handling as signal_handling,
    worker as worker,
)

r"""Utility classes & functions for data loading. Code in this folder is mostly used by ../dataloder.py.

A lot of multiprocessing is used in data loading, which only supports running
functions defined in global environment (py2 can't serialize static methods).
Therefore, for code tidiness we put these functions into different files in this
folder.
"""
IS_WINDOWS = ...
MP_STATUS_CHECK_INTERVAL = ...
python_exit_status = ...
HAS_NUMPY = ...
