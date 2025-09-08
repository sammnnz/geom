import numpy
import numpy as np

from abc import ABC
from functools import lru_cache
from geom.abc import BaseShape2D
from types import FunctionType, MethodType
from typing import cast
from .types import NDArray1D
from .utils import is_single_argument

__all__ = ["Circle", "Shape2D"]


class Shape2D(BaseShape2D, ABC):
    """ Parent class of all 2d shapes defined by arbitrary continuous curves. """

    def __init__(self, x, y, *args, phi=(0, 1), **kwargs):
        if not isinstance(x, (FunctionType, MethodType, np.ufunc)):
            raise TypeError("'x' must be a function.")

        if not is_single_argument(x):
            raise TypeError("'x' must have only one positional argument.")

        if not isinstance(y, (FunctionType, MethodType, np.ufunc)):
            raise TypeError("'y' must be a function.")

        if not is_single_argument(y):
            raise TypeError("'y' must have only one positional argument.")

        if not isinstance(phi, np.ndarray) or phi.dtype != np.float64:
            phi = cast(NDArray1D, np.asarray(phi, dtype=np.float64))  # may be ValueError with incorrect shape

        if phi.shape != (2,):
            raise ValueError("'phi' must be array-like object of shape (2,).")

        phi_0, phi_1 = phi
        if phi_0 >= phi_1:
            raise ValueError("'phi_0' must be < 'phi_1'.")

        x = x if isinstance(x, np.ufunc) else np.frompyfunc(x, 1, 1)
        y = y if isinstance(y, np.ufunc) else np.frompyfunc(y, 1, 1)
        # TODO: check the closure of curves
        self._x = x
        self._y = y
        self._phi = phi

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def phi(self):
        return self._phi


class Circle(Shape2D, ABC):
    """ Circle representation class. """

    def __init__(self, radius, center):
        if not isinstance(radius, np.number) or radius.dtype != np.float64:
            try:
                radius = np.float64(radius)
            except ValueError:
                raise ValueError("'radius' must be an integer or float.")

        if radius <= 0:
            raise ValueError("'radius' must be > 0.")

        if not isinstance(center, np.ndarray) or center.dtype != np.float64:
            center = cast(NDArray1D, np.asarray(center, dtype=np.float64))  # may be ValueError with incorrect shape

        if center.shape != (2,):
            raise ValueError("'center' must be array-like object of shape (2,).")

        x_0, y_0 = center

        def cos(phi):
            return x_0 + radius * np.cos(phi)

        def sin(phi):
            return y_0 + radius * np.sin(phi)

        super().__init__(x=cos, y=sin, phi=(0.0, 2.0 * np.pi))
        self._center = center
        self._radius = radius

    @lru_cache
    def area(self):
        return self._radius * self._radius * numpy.pi

    @property
    def center(self):
        return self._center

    @property
    def radius(self):
        return self._radius
