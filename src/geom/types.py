import numpy as np

from typing import TypeAlias, Union

__all__ = ["NDArray1D", "Number"]

Number: TypeAlias = Union[int, float, np.number]
NDArray1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
