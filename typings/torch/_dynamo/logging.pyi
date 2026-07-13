from collections.abc import Callable

import logging

"""Logging utilities for Dynamo and Inductor.

This module provides specialized logging functionality including:
- Step-based logging that prepends step numbers to log messages
- Progress bar management for compilation phases
- Centralized logger management for Dynamo and Inductor components

The logging system helps track the progress of compilation phases and provides structured
logging output for debugging and monitoring.
"""
disable_progress = ...

def get_loggers() -> list[logging.Logger]: ...

_step_counter = ...
if not disable_progress:
    num_steps = ...
    pbar = ...

def get_step_logger(logger: logging.Logger) -> Callable[..., None]: ...
