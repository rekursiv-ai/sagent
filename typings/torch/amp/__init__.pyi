# Fix: replace import to use redundant-alias form for re-exports.
from .autocast_mode import (
    autocast as autocast,
    custom_bwd as custom_bwd,
    custom_fwd as custom_fwd,
    is_autocast_available as is_autocast_available,
)
from .grad_scaler import GradScaler as GradScaler
