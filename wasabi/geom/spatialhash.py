"""Largely taken from 

http://themonkeyproject.wordpress.com/2011/05/18/introduction-to-spatial-hashes/
"""
import math

__all__ = [
    'SpatialHash',
]

class SpatialHash(object):
    def __init__(self, cell_size=250.0):
        self.cell_size = float(cell_size)
        self.d = {}

    def _add(self, cell_coord, o):
        """Add the object o to the cell at cell_coord."""
        self.d.setdefault(cell_coord, set()).add(o)

    def _cells_for_rect(self, r):
        """Return a set of the cells into which r extends."""
        cells = set()
        cy = math.floor(r.b / self.cell_size)
        while (cy * self.cell_size) <= r.t:
            cx = math.floor(r.l / self.cell_size)
            while (cx * self.cell_size) <= r.r:
                cells.add((int(cx), int(cy)))
                cx += 1.0
            cy += 1.0
        return cells

    def add_rect(self, r, obj):
        """Add an object obj with bounds r."""
        cells = self._cells_for_rect(r)
        for c in cells:
            self._add(c, obj)

    def _remove(self, cell_coord, o):
        """Remove the object o from the cell at cell_coord."""
        cell = self.d[cell_coord]
        cell.remove(o)

        # Delete the cell from the hash if it is empty.
        if not cell:
            del(self.d[cell_coord])

    def remove_rect(self, r, obj):
        """Remove an object obj which had bounds r."""
        cells = self._cells_for_rect(r)
        for c in cells:
            self._remove(c, obj)

    def potential_intersection(self, r):
        """Get a set of all objects that potentially intersect obj."""
        cells = self._cells_for_rect(r)
        seen = set()
        for c in cells:
            for hit in self.d.get(c, ()):
                if hit not in seen:
                    yield hit
                    seen.add(hit)
