"""
Tests for geom.figures_2d module
"""
from collections import deque

import numpy as np
import pytest

from geom.figures_2d import Figure2D, Triangle


class _Figure2D(Figure2D):
    def area(self):
        pass


_verts_valid = [[0, 0], [1, 0], [0, 1]]


@pytest.mark.parametrize("verts, shape", [
    (np.array(_verts_valid), (len(_verts_valid), len(_verts_valid[0]))),
    (_verts_valid, (len(_verts_valid), len(_verts_valid[0]))),
    (tuple(_verts_valid), (len(_verts_valid), len(_verts_valid[0]))),
    (deque(_verts_valid), (len(_verts_valid), len(_verts_valid[0]))),
    (_verts_valid + _verts_valid, (2 * len(_verts_valid), len(_verts_valid[0]))),
])
def test_Figure2D___init___valid(verts, shape):
    fig = _Figure2D(verts)
    verts = fig.verts
    assert isinstance(verts, np.ndarray) and verts.shape == shape


_match_value_error_invalid_shape = (r"\'verts\' must be array-like object of size n \* 2\, but ([0-9]*) \* ([0-9]*) "
                                    r"were given\.")
_match_value_error_invalid_vertices_count = (r"\'verts\' must be array-like object of size n \* 2\, where n >= (["
                                             r"0-9]*)\.")


@pytest.mark.parametrize("verts, error, kwargs", [
    ([[0, 0], [1, 1]], ValueError, {"match": _match_value_error_invalid_vertices_count}),
    ([[0, 0, 0], [1, 1, 1]], ValueError, {"match": _match_value_error_invalid_shape}),
    ([[0, 0], [1, 1, 1]], ValueError, {}),  # numpy error
    ("12345", ValueError, {"match": _match_value_error_invalid_shape}),
    ([[0, 0, 0]], ValueError, {"match": _match_value_error_invalid_shape}),
])
def test_Figure2D___init___invalid(verts, error, kwargs):
    with pytest.raises(error, **kwargs):
        _Figure2D(verts)


@pytest.mark.parametrize("verts", [
    [[0, 0], [1, 0], [0, 1]],
    [[0, 0], [1.0, 1.0], [0, 1.0]],
])
def test_Triangle___init___valid(verts):
    tri = Triangle(verts)
    assert tri.area() != 0.0


_match_value_error_invalid_vertices_count = r"Triangle must have size 3 \* 2\, but ([0-9]*) \* 2 were given\."
_match_value_error_triangle_degenerate = r"Triangle (.|\n)* is degenerate\."


@pytest.mark.parametrize("verts, error, kwargs", [
    ([[0, 0], [1, 1], [2, 1], [3, 1]], ValueError, {"match": _match_value_error_invalid_vertices_count}),
    ([[0, 0], [1, 1], [2, 2]], ValueError, {"match": _match_value_error_triangle_degenerate}),
])
def test_Triangle___init___invalid(verts, error, kwargs):
    with pytest.raises(error, **kwargs):
        Triangle(verts)


@pytest.mark.parametrize("verts, area", [
    ([[0, 0], [1, 0], [0, 1]], 0.5),
    ([[0, 0], [-1, 0], [0, -1]], 0.5),
    ([[1, 2], [0, 2], [1, 1]], 0.5),
])
def test_Triangle_area(verts, area):
    tri = Triangle(verts)
    assert tri.area() == area
