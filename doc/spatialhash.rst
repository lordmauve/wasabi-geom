Spatial Hashes
==============

A spatial hash is one way of indexing objects in space. As Python programmers,
we should perhaps call them spatial dicts, but let’s go with the literature on
this one. Like a dict, a spatial hash has O(1) properties.

The basic principle is to split space into an infinite number of cells – each
cell can contain an arbitrary number of objects. A cell that is empty simply
isn’t stored.

.. automodule:: wasabi.geom.spatialhash
    
    .. autoclass:: SpatialHash
        :members:
