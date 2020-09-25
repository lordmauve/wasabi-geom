# wasabigeom - fast geometry types for Python games

![Build Wheels](https://github.com/lordmauve/cyvec/workflows/Build%20Wheels/badge.svg?branch=master)

`wasabigeom` is a 2D geometry library intended for game development. It started
life as a pure Python library but is now implemented in optimised Cython code.


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
