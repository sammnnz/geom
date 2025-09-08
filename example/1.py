import numpy as np

from functools import lru_cache
from geom.figures_2d import Figure2D


def is_rectangle(verts):
    """ Return True if vertices define a rectangle, else return False. """
    matrix = np.concat((verts[1:] - verts[:-1], [verts[0] - verts[-1]]))
    prod = np.sum(matrix * np.roll(matrix, 1, axis=0), axis=1)
    return np.all(prod == 0.0)


class Rectangle(Figure2D):
    """ An example of how you can easily add a rectangle based on Figure2D. """

    def __init__(self, verts, *args, **kwargs):
        super(Rectangle, self).__init__(verts, *args, **kwargs)
        n = len(self._verts)
        if n != 4:
            raise ValueError("Rectangle must have size 4 * 2, but %i * 2 were given." % n)

        if not is_rectangle(self._verts):
            raise ValueError("Rectangle must have only right angles.")

        if self.area() == 0.0:
            raise ValueError("Rectangle %s is degenerate." % self)

    @lru_cache
    def area(self):
        matrix = self._verts[1:3] - self._verts[:2]
        return np.linalg.norm(matrix[0]) * np.linalg.norm(matrix[1])


if __name__ == "__main__":
    verts = (np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) @
             np.array([
                 [np.cos(np.pi / 4), np.sin(np.pi / 4)],
                 [-np.sin(np.pi / 4), np.cos(np.pi / 4)]
             ]))
    rect = Rectangle(verts)
    assert rect.area() == 4.0
