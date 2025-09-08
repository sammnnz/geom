import numpy as np

from abc import ABC
from functools import lru_cache
from geom.abc import BaseFigure2D

__all__ = ["Figure2D", "Triangle"]


class Figure2D(BaseFigure2D, ABC):
    """ Parent class of all 2d figures with a finite number of vertices. """

    def __init__(self, verts, *args, **kwargs):
        if not isinstance(verts, np.ndarray) or verts.dtype != np.float64:
            verts = np.asarray(verts, dtype=np.float64)  # may be ValueError with incorrect shape

        n, m = 0, 0
        try:
            n, m = verts.shape
            if m != 2:
                raise ValueError
        except ValueError:
            raise ValueError("'verts' must be array-like object of size n * 2, but %i * %i were given." % (n, m))

        if n < 3:
            raise ValueError("'verts' must be array-like object of size n * 2, where n >= 3.")

        self._verts = verts

    def __str__(self):
        return str(self._verts)

    @property
    def verts(self):
        return self._verts


class Triangle(Figure2D, ABC):
    """ Triangle representation class. """

    def __init__(self, verts, *args, **kwargs):
        super().__init__(verts, *args, **kwargs)
        n = len(verts)
        if n != 3:
            raise ValueError("Triangle must have size 3 * 2, but %i * 2 were given." % n)

        if self.area() == 0.0:
            raise ValueError("Triangle %s is degenerate." % self)

    @lru_cache
    def area(self):
        matrix = self._verts[1:] - self._verts[0]
        det = np.linalg.det(matrix)
        return np.abs(det) / 2.0

    @lru_cache
    def is_right_angled(self):
        matrix = np.concat((self._verts[1:] - self._verts[:-1], [self._verts[0] - self._verts[-1]]))
        prod = np.sum(matrix * np.roll(matrix, 1, axis=0), axis=1)
        return 0.0 in prod
