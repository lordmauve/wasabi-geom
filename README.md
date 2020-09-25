# wasabigeom - fast geometry types for Python games

![Build Wheels](https://github.com/lordmauve/cyvec/workflows/Build%20Wheels/badge.svg?branch=master)
![PyPI](https://img.shields.io/pypi/v/wasabi-geom) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/wasabi-geom) ![PyPI - Wheel](https://img.shields.io/pypi/wheel/wasabi-geom) [![Documentation Status](https://readthedocs.org/projects/wasabigeom/badge/?version=stable)](https://wasabigeom.readthedocs.io/en/stable/?badge=stable) [![Discord](https://img.shields.io/discord/705530610847973407)](https://discord.gg/jBWaWHU)

`wasabigeom` is a 2D geometry library intended for game development. It started
life as a pure Python library but is now implemented in optimised Cython code.

# Documentation

[View on ReadTheDocs](https://wasabigeom.readthedocs.io/en/stable/)


# Installation

To install, just run:

```
pip install wasabi-geom
```


## What's new in 2.0.0

I took the existing `wasabi.geom` code and Cythonised it.

I've made some big, breaking changes to the interface; notably, I prefer
radians thes days and eschew namespace packages. To install the old,
pure-Python version, pin to `wasabi-geom<2`.
