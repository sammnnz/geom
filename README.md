# Geom

<a><img src="https://img.shields.io/badge/python-3.10 | 3.11 | 3.12 | 3.13 | 3.14 -blue.svg"></a>
![Tests](https://github.com/sammnnz/geom/actions/workflows/tests-win.yml/badge.svg)
![Tests](https://github.com/sammnnz/geom/actions/workflows/tests-ub.yml/badge.svg)
![Tests](https://github.com/sammnnz/geom/actions/workflows/tests-mac.yml/badge.svg)
[![codecov](https://codecov.io/gh/sammnnz/geom/branch/main/graph/badge.svg?token=BlyE68KvAS)](https://codecov.io/gh/sammnnz/geom)

Демонстрационная библиотека реализующая классы-представления двумерных фигур и вычисление их площадей.

## Install
~~~~shell
pip install "git+https://github.com/sammnnz/geom@v0.2.0"
~~~~
или
~~~~shell
poetry add "git+https://github.com/sammnnz/geom@v0.2.0"
~~~~

## Features
**BaseFigure2D**, **BaseShape2D** - абстрактные классы модуля [geom.abc](https://github.com/sammnnz/geom/blob/main/src/geom/abc.py) требующие реализации метода `area`.

**Figure2D** - родительский класс (модуля [geom.figures_2d](https://github.com/sammnnz/geom/blob/main/src/geom/figures_2d.py)) для n-угольника, где $n \in \mathbb{N}:\ n \geq 3$ .

**Triangle** - класс наследованный от **Figure2D**, представляющий треугольник.

**Shape2D** - родительский класс (модуля [geom.shapes_2d](https://github.com/sammnnz/geom/blob/main/src/geom/shapes_2d.py)) для произвольного многоугольника, чья граница заданна непрерывной спрямляемой замкнутой кривой $\gamma (\varphi): [\varphi_0,\ \varphi_1] \rightarrow\ \mathbb{R}^2$.

**Circle** - класс наследованный от **Shape2D**, представляющий круг.

Вычислить площадь (для любой из фигур) можно вызвав метод `area`.

Узнать прямоугольный треугольник или нет, можно вызвав метод `is_right_angled` класса `Triangle`.

Добавление дополнительных классов показано в примере [1](https://github.com/sammnnz/geom/blob/main/example/1.py).

## Tests
Протестировано для `CPython` `3.10`, `3.11`, `3.12`, `3.13`, `3.14-rc` на `Windows`, `Linux`, `Mac`.
