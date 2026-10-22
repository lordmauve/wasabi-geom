API
===

.. currentmodule:: wasabigeom

Vectors
-------

.. autoclass:: wasabigeom.vec2
    :members:

    ``vec2`` is a 2D vector that supports standard mathematical operations like
    addition and multiplication:

    .. code-block:: pycon

        >>> vec2(3, 5) + (1, 2)
        vec2(4.0, 7.0)
        >>> vec2(1, 2) * 0.5
        vec2(0.5, 1.0)

    ``vec2`` is immutable and hashable.


Transformations
---------------

.. autoclass:: wasabigeom.Transform
    :members: __init__, build, __mul__, transform, inverse, factorise


.. autoclass:: wasabigeom.Matrix

    .. deprecated:: 2.1.0

        Use :class:`wasabigeom.Transform` instead.


Geometric Primitives
--------------------

.. autoclass:: wasabigeom.Polygon
.. autoclass:: wasabigeom.ConvexPolygon
.. autoclass:: wasabigeom.Triangle
.. autoclass:: wasabigeom.Line
.. autoclass:: wasabigeom.Segment
.. autoclass:: wasabigeom.PolyLine
.. autoclass:: wasabigeom.Projection


Polygon triangulation
---------------------

.. autofunction:: wasabigeom.triangulate

The adapter uses vendored `Earstrip <https://github.com/GLMap/earstrip>`_
(C++17, ISC licensed). It returns indices into the original vertices, never
new intersection points. It does not expose Earstrip's constrained/approximate
repair modes. The caller is responsible for supplying simple rings; failed
exact triangulation raises ``ValueError``.

``vertices`` must export a native-endian float32 or float64 two-dimensional
buffer. NumPy is optional. For example, using only the standard library::

    from array import array
    from wasabigeom import triangulate

    xy = array('d', [0, 0, 10, 0, 10, 10, 0, 10,
                     3, 3, 7, 3, 7, 7, 3, 7])
    vertices = memoryview(xy).cast('B').cast('d', (8, 2))
    indices = triangulate(vertices, ring_ends=[4, 8])

Ring ends are cumulative counts, not hole starts. Omit them for a single ring.
The returned ``array('I')`` owns its storage and remains valid after later calls
and after the input is released. With NumPy, ``numpy.asarray(indices)`` obtains
a zero-copy one-dimensional uint32 view. Triangle order and diagonals may differ
from other triangulators. Areas and winding, rather than index equality, should
be used when comparing valid results.

Building from source now also requires a C++17 compiler. The original
``wasabigeom`` extension remains C; only the separate Earstrip extension uses
C++17. Windows/MSVC uses ``/std:c++17``; other supported toolchains use
``-std=c++17``. Linux build/distribution tests have been run locally; other
platforms rely on the existing wheel CI.


Bounds and collision tests
--------------------------

.. autoclass:: wasabigeom.Rect
.. autoclass:: wasabigeom.SpatialHash


Rasterisation
-------------

.. autofunction:: wasabigeom.bresenham
.. autofunction:: wasabigeom.quadratic_bezier
.. autofunction:: wasabigeom.cubic_bezier
.. autofunction:: wasabigeom.circular_arc

Axis-Aligned, pygame-compatible Rect
------------------------------------

.. autoclass:: wasabigeom.ZRect
