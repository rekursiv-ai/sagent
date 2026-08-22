from typing import Any

from numpy import ndarray
from numpy.typing import NDArray
from torch import dtype

def figure_to_image(figures, close=...) -> NDArray[Any]: ...
def make_grid(I, ncols=...) -> ndarray[tuple[int, int, int], dtype[Any]]: ...
def convert_to_HWC(
    tensor, input_format
) -> ndarray[tuple[int, int, int], dtype[Any]] | NDArray[Any] | None: ...
