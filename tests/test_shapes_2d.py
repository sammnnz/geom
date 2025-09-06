"""
Tests for geom.shapes_2d module
"""
import numpy as np
import pytest

from geom.shapes_2d import Circle, Shape2D


class _Shape2D(Shape2D):
    def area(self):
        pass


def x_valid(phi):
    return np.cos(phi)


def y_valid(phi):
    return np.sin(phi)


# TODO: more tests
@pytest.mark.parametrize("x, y, phi", [
    (x_valid, y_valid, (0, 2 * np.pi)),
])
def test_Shape2D___init___valid(x, y, phi):
    sh = _Shape2D(x, y, phi=phi)
    assert isinstance(sh.x, np.ufunc)
    assert isinstance(sh.y, np.ufunc)
    assert np.array_equal(sh.phi, phi)


def x_invalid():
    pass


def y_invalid(phi, psi=1):
    pass


@pytest.mark.parametrize("x, y, phi, error", [
    (x_invalid, y_valid, (0, 2 * np.pi), TypeError),
    (x_valid, y_invalid, (0, 2 * np.pi), TypeError),
    (None, y_valid, (0, 2 * np.pi), TypeError),
    (x_valid, None, (0, 2 * np.pi), TypeError),
    (x_valid, y_valid, (0, 1, 2), ValueError),
])
def test_Shape2D___init___invalid(x, y, phi, error):
    with pytest.raises(error):
        _Shape2D(x, y, phi=phi)


# TODO: more tests
@pytest.mark.parametrize("radius, center", [
    (1, (0, 0)),
])
def test_Circle___init___valid(radius, center):
    ci = Circle(radius, center)
    assert isinstance(ci.x, np.ufunc) and isinstance(ci.y, np.ufunc)


# TODO: more tests
@pytest.mark.parametrize("radius, center, area", [
    (1, (0, 0), np.pi),
])
def test_Circle_area(radius, center, area):
    ci = Circle(radius, center)
    assert ci.area() == area
