import numpy
import numpy as np

from abc import ABC
from functools import lru_cache
from geom.abc import BaseShape2D
from types import FunctionType, MethodType
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

        try:
            n, m = phi
        except ValueError:
            raise ValueError("'phi' must be array-like object of size 2 * 1.")

        if not isinstance(n, (int, float, np.number)) or not isinstance(m, (int, float, np.number)):
            raise ValueError("'phi' must have an integer or float.")

        self._x = x if isinstance(x, np.ufunc) else np.frompyfunc(x, 1, 1)
        self._y = y if isinstance(y, np.ufunc) else np.frompyfunc(y, 1, 1)
        self._phi = phi if isinstance(phi, np.ndarray) else np.array([n, m])

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
        if not isinstance(radius, (int, float, np.number)) or radius <= 0:
            raise ValueError("'radius' must be an integer or float and > 0.")

        try:
            x_0, y_0 = center
        except ValueError:
            raise ValueError("'center' must be an array with length 2.")

        if not isinstance(x_0, (int, float, np.number)) or not isinstance(y_0, (int, float, np.number)):
            raise ValueError("'center' coordinates must be an integer or float.")

        def cos(phi):
            return x_0 + radius * np.cos(phi)

        def sin(phi):
            return y_0 + radius * np.sin(phi)

        super(Circle, self).__init__(x=cos, y=sin, phi=(0.0, 2.0 * np.pi))
        self._radius = radius
        self._center = center

    @lru_cache
    def area(self):
        return self._radius * self._radius * numpy.pi

    @property
    def center(self):
        return self._center

    @property
    def radius(self):
        return self._radius
