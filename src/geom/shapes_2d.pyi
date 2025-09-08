import numpy as np
import numpy.typing as npt

from abc import ABC
from geom.abc import BaseShape2D
from typing import Callable, List, Sequence, Union
from .types import Number, NDArray1D

__all__: List[str]

class Shape2D(BaseShape2D, ABC):
    _x: np.ufunc
    _y: np.ufunc
    _phi: NDArray1D
    def __init__(self,
                 x: Callable[[Number], Number],
                 y: Callable[[Number], Number], *args,
                 phi: Union[Sequence, npt.NDArray] =..., **kwargs) -> None: ...
    def area(self) -> np.float64: ...
    @property
    def x(self) -> np.ufunc: ...
    @property
    def y(self) -> np.ufunc: ...
    @property
    def phi(self) -> NDArray1D: ...

class Circle(Shape2D, ABC):
    _center: NDArray1D
    _radius: np.float64
    def __init__(self, radius: Number, center: Union[Sequence, npt.NDArray]) -> None: ...
    def area(self) -> np.float64: ...
    @property
    def center(self) -> NDArray1D: ...
    @property
    def radius(self) -> np.float64: ...
