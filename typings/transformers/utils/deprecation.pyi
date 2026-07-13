from . import ExplicitEnum

class Action(ExplicitEnum):
    NONE = ...
    NOTIFY = ...
    NOTIFY_ALWAYS = ...
    RAISE = ...

def deprecate_kwarg(
    old_name: str,
    version: str,
    new_name: str | None = ...,
    warn_if_greater_or_equal_version: bool = ...,
    raise_if_greater_or_equal_version: bool = ...,
    raise_if_both_names: bool = ...,
    additional_message: str | None = ...,
):  # -> Callable[..., _Wrapped[..., Any, ..., Any]]:
    ...
