import weakref

_SKIP_TYPES = ...

def find_legit_leaks_from_referrers(
    active_fakes: weakref.WeakSet,
) -> weakref.WeakSet: ...
