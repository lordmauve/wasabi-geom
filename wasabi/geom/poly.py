import math

from collections import namedtuple


from .vector import Vector, cached
from .lines import Projection, PolyLine, LineSegment


__all__ = [
    'Rect',
    'Triangle',
    'ConvexPolygon',
    'Polygon'
]


class BasePolygon(object):
    """Base polygon.

    Operations defined on a set of points.
    """

    def translate(self, v):
        """Return a translated instance of this polygon."""
        return self.__class__(
            *(p + v for p in self)
        )

    def __iter__(self):
        return iter(self.points)

    def project_to_axis(self, axis):
        """Project the polygon onto the vector axis, which must be normalised.
        """
        projected_points = [p.dot(axis) for p in self.points]
        return Projection(min(projected_points), max(projected_points))
  
    def intersects(self, other):
        """Determine if this convex polygon intersects another convex polygon.

        Returns False if there is no intersection, or a separation vector if
        there is an intersection.

        """
        edges = self.edges
        edges.extend(other.edges)
        
        projections = []
        for edge in edges:
            axis = edge.normalised().perpendicular()
            
            self_projection = self.project_to_axis(axis)
            other_projection = other.project_to_axis(axis)
            intersection1 = self_projection.intersection(other_projection)
            intersection2 = -other_projection.intersection(self_projection)
            if not intersection1:
                return False
                
            proj_vector1 = Vector((axis.x * intersection1, axis.y * intersection1))
            proj_vector2 = Vector((axis.x * intersection2, axis.y * intersection2))
            projections.append(proj_vector1)
            projections.append(proj_vector2)
        
        mtd = -self.find_mtd(projections)
        
        return mtd
    
    def find_mtd(self, push_vectors):
        mtd = push_vectors[0]
        mind2 = push_vectors[0].dot(push_vectors[0])
        for vector in push_vectors[1:]:
            d2 = vector.dot(vector)
            if d2 < mind2:
                mind2 = d2
                mtd = vector
        return mtd


class ConvexPolygon(BasePolygon):
    def __init__(self, points):
        self.points = []
        for p in points:
            self.points.append(Vector(p))
        
    @cached
    def edges(self):
        edges = []
        for i in xrange(len(self.points)):
            point = self.points[i]
            next_point = self.points[(i + 1) % len(self.points)]
            edges.append(next_point - point)
        return edges

    def to_tri_strip(self):
        """Generate a list of the points in triangle-strip order"""
        left = 0
        right = len(self.points) - 1
        while True:
            yield self.points[left]
            yield self.points[right]
            
            left += 1
            right -= 1
            if left == right:
                yield self.points[left]
            elif left > right:
                break

    def segments(self):
        nvs = len(self.points)
        for i in xrange(nvs):
            v1 = i
            v2 = (i + 1) % nvs
            yield LineSegment(self.points[v1], self.points[v2])


class Triangle(object):
    """Two-dimensional vector (oriented) triangle implementation.

    """

    def __init__(self, base, primary, secondary):
        """Create a Triangle object.

        The two vectors that define the edges from the base point are ordered.
        If the vectors are counter-clockwise and the triangle is considered
        counter-clockwise, ditto clockwise.

        :Parameters:
            `base` : Vector
                A point vector of a base point of the triangle.
            `primary` : Vector
                The vector from the base point to one of the others.
            `secondary` : Vector
                The vector from the base point to the final point.

        """
        if not isinstance(base, Vector):
            base = Vector(base)
        if not isinstance(primary, Vector):
            primary = Vector(primary)
        if not isinstance(secondary, Vector):
            secondary = Vector(secondary)
        self.base = base
        self.primary = primary
        self.secondary = secondary

    def __str__(self):
        """Construct a concise string representation.

        """
        params = (self.base, self.primary, self.secondary)
        return "Triangle(%s, %s, %s)" % params

    def __repr__(self):
        """Construct a precise string representation.

        """
        params = (self.base, self.primary, self.secondary)
        return "Triangle(%r, %r, %r)" % params

    @classmethod
    def from_points(cls, base, first, second):
        """Create a Triangle object from its three points.

        :Parameters:
            `base` : Vector
                The base point of the triangle.
            `first`, `second` : Vector
                The other two points of the triangle.

        """
        if not isinstance(base, Vector):
            base = Vector(base)
        primary = first - base
        secondary = second - base
        return cls(base, primary, secondary)

    @cached
    def area(self):
        """The unsigned area of the triangle.

        """
        area = self.primary.cross(self.secondary) / 2
        self.is_clockwise = (area < 0)
        return abs(area)

    @cached
    def is_clockwise(self):
        """True if the primary and secondary are clockwise.

        """
        area = self.primary.cross(self.secondary) / 2
        self.area = abs(area)
        return (area < 0)

    @cached
    def first(self):
        """The point at the end of the primary vector.

        """
        return self.base + self.primary

    @cached
    def second(self):
        """The point at the end of the secondary vector.

        """
        return self.base + self.secondary


class Rect(BasePolygon, namedtuple('BaseRect', 'l r b t')):
    """2D rectangle class."""    

    @classmethod
    def from_blwh(cls, bl, w, h):
        """Construct a Rect from its bottom left and dimensions."""
        l, b = bl
        return Rect(
            l,
            l + w,
            b,
            b + h
        )

    @classmethod
    def from_cwh(cls, c, w, h):
        """Construct a Rect from its center and dimensions."""
        w2 = w * 0.5
        h2 = h * 0.5
        return cls(
            c.x - w2,
            c.x + w2,
            c.y - h2,
            c.y + h2
        )

    @classmethod
    def from_points(cls, p1, p2):
        """Construct the smallest Rect that contains the points p1 and p2."""
        x1, y1 = p1
        x2, y2 = p2
        if x2 < x1:
            x1, x2 = x2, x1

        if y2 < y1:
            y1, y2 = y2, y1

        return cls(
            x1,
            x2,
            y1,
            y2
        )

    @classmethod
    def as_bounding(cls, points):
        """Construct a Rect as the bounds of a sequence of points.

        :param points: An iterable of the points to bound.

        """
        xs, ys = zip(*points)
        lo = (min(xs), min(ys))
        hi = (max(xs), max(ys))
        return cls(lo, hi)

    @property
    def points(self):
        """A list of the points in the rectangle."""
        return [
            Vector((self.l, self.b)),
            Vector((self.l, self.t)),
            Vector((self.r, self.t)),
            Vector((self.r, self.b)),
        ]

    @property
    def edges(self):
        edges = []
        points = self.points
        last = points[-1]
        for i, p in enumerate(points):
            edges.append(p - last)
            last = p
        return edges

    def contains(self, p):
        """Return True if the point p is within the Rect."""
        x, y = p
        return (
            self.l <= x < self.r and
            self.b <= y < self.t
        )

    def overlaps(self, r):
        """Return True if this Rect overlaps another.
        
        Not to be confused with .intersects(), which works for arbitrary convex
        polygons and computes a separation vector.

        """
        return (
            r.r > self.l and r.l < self.r and
            r.t > self.b and r.b < self.t
        )

    def intersection(self, r):
        """The intersection of this Rect with another."""
        if not self.overlaps(r):
            return None
        xs = [self.l, self.r, r.l, r.r]
        ys = [self.b, self.t, r.b, r.t]
        xs.sort()
        ys.sort()
        return Rect(xs[1], ys[1], xs[2] - xs[1], ys[2] - ys[1])

    @property
    def w(self):
        """Width of the rectangle."""
        return self.r - self.l

    @property
    def h(self):
        """Height of the rectangle."""
        return self.t - self.b

    def bottomleft(self):
        """The bottom left point."""
        return Vector((self.l, self.b))

    def topleft(self):
        """The top left point."""
        return Vector((self.l, self.t))
    
    def topright(self):
        """The top right point."""
        return Vector((self.r, self.t))

    def bottomright(self):
        """The bottom right point."""
        return Vector((self.r, self.b))

    def translate(self, off):
        """Return a new Rect translated by the vector `off`."""
        x, y = off
        return Rect(
            self.l + x,
            self.r + x,
            self.b + y,
            self.t + y
        )

    def __repr__(self):
        """A string representation of the Rect."""
        return "Rect(%r, %r, %r, %r)" % self


class Polygon(object):
    """Mutable polygon, possibly with holes, multiple contours, etc.

    This exists mainly as a wrapper for polygon triangulation, but also
    provides some useful methods.

    """
    def __init__(self, vertices=None):
        self.contours = []
        if vertices:
            self.add_contour(vertices)

    def mirror(self, plane):
        p = Polygon()
        for c in self.contours:
            mirrored = [plane.mirror(v) for v in reversed(c)]
            p.add_contour(mirrored)
        return p

    def polylines_facing(self, v, threshold=0):
        """Compute a list of PolyLines on the edge of this contour whose normals face v.
        
        threshold the value of the segment normal dot v required to include
        a segment in the polyline.
        
        """
        lines = []
        for contour in self.contours:
            # first work out which segments pass
            segments = []
            nvs = len(contour)
            for i in range(nvs):
                v1 = i
                v2 = (i + 1) % nvs
                segment = LineSegment(contour[v1], contour[v2])
                try:
                    normal = segment.normal()
                except ZeroDivisionError:
                    continue
                facing = segment.normal().dot(v) > threshold
                segments.append((segment, facing))

            nvs = len(segments)

            # find a non-facing/facing boundary to start
            was_facing = None
            for start in range(nvs):
                facing = segments[start][1]
                if was_facing is None:
                    was_facing = facing
                elif was_facing ^ facing:
                    break

            # 'start' is now an offset we can start at to find all connected segments
            vs = []
            for i in range(nvs):
                seg, facing = segments[(i + start) % nvs]
                if not facing:
                    if vs:
                        lines.append(vs)
                        vs = []
                else:
                    if vs:
                        vs.append(seg.p2)
                    else:
                        vs = [seg.p1, seg.p2]
            if vs:
                lines.append(vs)
        return [PolyLine(vs) for vs in lines if len(vs) >= 2]
        

    def add_contour(self, vertices):
        """Adds a contour"""
        self.contours.append(vertices)
