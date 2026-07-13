from ._normalizations import ArrayLike, normalizer

"""Wrapper to mimic (parts of) np.random API surface.

NumPy has strict guarantees on reproducibility etc; here we don't give any.

Q: default dtype is float64 in numpy

"""
__all__ = [
    "choice",
    "normal",
    "rand",
    "randint",
    "randn",
    "random",
    "random_sample",
    "sample",
    "seed",
    "shuffle",
    "uniform",
]

def use_numpy_random() -> bool: ...
def deco_stream(func) -> _Wrapped[..., Any, ..., Any | ndarray]: ...
@deco_stream
def seed(seed=...) -> None: ...
@deco_stream
def random_sample(size=...) -> float | ndarray: ...
def rand(*size) -> float | ndarray: ...

sample = ...
random = ...

@deco_stream
def uniform(low=..., high=..., size=...) -> float | ndarray: ...
@deco_stream
def randn(*size) -> float | ndarray: ...
@deco_stream
def normal(loc=..., scale=..., size=...) -> float | ndarray: ...
@deco_stream
def shuffle(x) -> None: ...
@deco_stream
def randint(low, high=..., size=...) -> float | ndarray: ...
@deco_stream
@normalizer
def choice(a: ArrayLike, size=..., replace=..., p: ArrayLike | None = ...): ...
