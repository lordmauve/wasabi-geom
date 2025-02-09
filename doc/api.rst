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
