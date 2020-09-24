#cython: language_level=3, c_api_binop_methods=True

cimport cython
from libc.math cimport sqrt, atan2, cos, sin, floor, pi
from libc.stdlib cimport llabs
from libc.stdint cimport int64_t, uint64_t


cdef inline int _extract(object o, double *x, double *y):
    if isinstance(o, vec2):
        x[0] = (<vec2> o).x;
        y[0] = (<vec2> o).y;
        return 1
    if len(o) == 2:
        x[0] = <double?> o[0];
        y[0] = <double?> o[1];
        return 1
    return 0


cdef vec2 newvec2(double x, double y):
    cdef vec2 v = vec2.__new__(vec2)
    v.x = x
    v.y = y
    return v


@cython.freelist(32)
cdef class vec2:
    """Two-dimensional float vector implementation."""
    cdef readonly double x, y

    def __init__(self, *args):
        if len(args) == 2:
            self.x = <double?> args[0]
            self.y = <double?> args[1]
            return
        elif len(args) == 1:
            if _extract(args[0], &self.x, &self.y):
                return
        raise TypeError(
            "Expected a vector object or tuple, or x and y parameters"
        )

    cdef object stringify(self):
        return f"vec2({self.x!r}, {self.y!r})"

    def __str__(self):
        """Construct a concise string representation."""
        return self.stringify()

    def __repr__(self):
        """Construct a precise string representation."""
        return self.stringify()

    def __len__(self):
        return 2

    def __eq__(self, other):
        cdef double x2, y2
        cdef bint res
        if not _extract(other, &x2, &y2):
            return NotImplemented
        res = self.x == x2 and self.y == y2
        return res

    def __hash__(self):
        cdef uint64_t hash, h2, mask = 0xffffffff, shift = 32
        hash = (<uint64_t *> &self.x)[0]
        h2 = (<uint64_t *> &self.y)[0]
        hash = hash ^ (h2 << shift) | (h2 >> shift & mask)
        return hash

    def __getitem__(self, int idx):
        if idx == 0:
            return self.x
        elif idx == 1:
            return self.y
        raise IndexError(idx)

    cpdef double length(self):
        """Return length of the vector."""
        return sqrt(self.x * self.x + self.y * self.y)

    def __abs__(self):
        return self.length()

    cpdef double length_squared(self):
        """Return the square of the length of the vector."""
        return self.x * self.x + self.y * self.y

    cpdef double angle(self):
        """The angle the vector makes to the positive x axis, in radians."""
        return atan2(self.y, self.x)

    cpdef bint is_zero(self):
        """Test if this is the zero vector."""
        return self.length_squared() < 1e-9

    def __add__(object a, object b):
        """Add the vectors componentwise.

        :Parameters:
            `other` : vec2
                The object to add.

        """
        cdef double ax, ay, bx, by
        if not _extract(a, &ax, &ay) or not _extract(b, &bx, &by):
            return NotImplemented
        return newvec2(ax + bx, ay + by)

    def __sub__(object a, object b):
        """Subtract the vectors componentwise."""
        cdef double ax, ay, bx, by
        if not _extract(a, &ax, &ay) or not _extract(b, &bx, &by):
            return NotImplemented
        return newvec2(ax - bx, ay - by)

    def __mul__(vec2 self, other):
        """Either multiply the vector by a scalar or compute the dot product
        with another vector.

        :Parameters:
            `other` : vec2 or float
                The object by which to multiply.

        """
        cdef double v
        if isinstance(other, float):
            v = <double> other
            return newvec2(self.x * v, self.y * v)
        return self.dot(other)

    def __div__(self, other):
        """Divide the vector by a scalar.

        :Parameters:
            `other` : float
                The object by which to divide.

        """
        return vec2(self.x / other, self.y / other)

    def __truediv__(self, other):
        """Divide the vector by a scalar.

        :Parameters:
            `other` : float
                The object by which to divide.

        """
        return vec2(self.x / other, self.y / other)

    def __floordiv__(self, other):
        """Divide the vector by a scalar, rounding down.

        :Parameters:
            `other` : float
                The object by which to divide.

        """
        return vec2(self.x // other, self.y // other)

    def __neg__(self):
        """Compute the unary negation of the vector.

        """
        return vec2(-self.x, -self.y)

    def rotated(self, angle):
        """Compute the vector rotated by an angle.

        :Parameters:
            `angle` : float
                The angle (in radians) by which to rotate.

        """
        vx = self.x
        vy = self.y
        ca, sa = cos(angle), sin(angle)
        return vec2((vx * ca - vy * sa, vx * sa + vy * ca))

    def scaled_to(self, length):
        """Compute the vector scaled to a given length.

        :Parameters:
            `length` : float
                The length to which to scale.

        """
        vx = self.x
        vy = self.y
        s = length / self.length()
        return vec2(vx * s, vy * s)

    def safe_scaled_to(self, length):
        """Compute the vector scaled to a given length, or just return the
        vector if it was the zero vector.

        :Parameters:
            `length` : float
                The length to which to scale.

        """
        if self.is_zero():
            return self
        return self.scaled_to(length)

    def normalized(self):
        """Compute the vector scaled to unit length.

        """
        cdef double l = self.length()
        return newvec2(self.x / l, self.y / l)

    def safe_normalized(self):
        """Compute the vector scaled to unit length, or some unit vector
        if it was the zero vector.

        """
        if self.is_zero():
            return vec2(0, 1)
        return self.normalized()

    def perpendicular(self):
        """Compute the perpendicular."""
        vx, vy = self
        return newvec2(-vy, vx)

    cpdef double dot(vec2 self, object other):
        """Compute the dot product with another vector.

        :Parameters:
            `other` : vec2
                The vector with which to compute the dot product.

        """
        cdef double x2, y2
        if not _extract(other, &x2, &y2):
            raise TypeError("Expected vec2 or 2-tuple")
        return self.x * x2 + self.y * y2

    def cross(self, other):
        """Compute the cross product with another vector.

        :Parameters:
            `other` : vec2
                The vector with which to compute the cross product.

        """
        return self[0] * other[1] - self[1] * other[0]

    def project(self, other):
        """Compute the projection of another vector onto this one.

        :Parameters:
            `other` : vec2
                The vector of which to compute the projection.

        """
        return self * self.dot(other) / self.dot(self)

    def angle_to(self, other):
        """Compute the angle made to another vector in the range [0, pi].

        :Parameters:
            `other` : vec2
                The vector with which to compute the angle.

        """
        if not isinstance(other, vec2):
            other = vec2(other)
        a = abs(other.angle() - self.angle())
        return min(a, 2 * pi - a)

    def signed_angle_to(self, other):
        """Compute the signed angle made to another vector in the range.

        :Parameters:
            `other` : vec2
                The vector with which to compute the angle.

        """
        if not isinstance(other, vec2):
            other = vec2(other)
        a = other.angle() - self.angle()
        return min(a + pi, a, a - pi, key=abs)

    def to_polar(self):
        return self.length(), self.angle()

    def distance_to(self, other):
        """Compute the distance to another point vector.

        :Parameters:
            `other` : vec2
                The point vector to which to compute the distance.

        """
        return (other - self).length()

def v(*args):
    """Construct a vector from an iterable or from multiple arguments. Valid
    forms are therefore: ``v((x, y))`` and ``v(x, y)``.

    """
    if len(args) == 2:
        x, y = args
    elif len(args) == 1:
        x, y = args[0]
    else:
        raise TypeError(
            "Expected either a two-argument tuple or two arguments"
        )
    return vec2(x, y)


#: The zero vector.
zero = vec2(0, 0)

#: The unit vector on the x-axis.
unit_x = vec2(1, 0)

#: The unit vector on the y-axis.
unit_y = vec2(0, 1)


class Line(object):
    """Two-dimensional vector (directed) line implementation.

    Lines are defined in terms of a perpendicular vector and the distance from
    the origin.

    The representation of the line allows it to partition space into an
    'outside' and an inside.

    """

    def __init__(self, direction, distance):
        """Create a Line object.

        :Parameters:
            `direction` : vec2
                A (non-zero) vector perpendicular to the line.
            `distance` : float
                The distance from the origin to the line.

        """
        if not isinstance(direction, vec2):
            direction = vec2(direction)
        self.direction = direction.normalized()
        self.along = self.direction.perpendicular()
        self.distance = distance

    def __neg__(self):
        """Return the opposite of this Line.

        The opposite Line represents the same line through space but partitions
        it in the opposite direction.
        """
        return Line(-self.direction, -self.distance)

    def __str__(self):
        """Construct a concise string representation.

        """
        return "Line(%s, %.2f)" % (self.direction, self.distance)

    def __repr__(self):
        """Construct a precise string representation.

        """
        return "Line(%r, %r)" % (self.direction, self.distance)

    @classmethod
    def from_points(cls, first, second):
        """Create a Line object from two (distinct) points.

        :Parameters:
            `first`, `second` : vec2
                The vectors used to construct the line.

        """
        if not isinstance(first, vec2):
            first = vec2(first)
        along = (second - first).normalized()
        direction = -along.perpendicular()
        distance = first.dot(direction)
        return cls(direction, distance)

    def offset(self):
        """The projection of the origin onto the line.

        """
        return self.direction * self.distance

    def project(self, point):
        """Compute the projection of a point onto the line.

        :Parameters:
            `point` : vec2
                The point to project onto the line.

        """
        parallel = self.along.project(point)
        return parallel + self.offset

    def reflect(self, point):
        """Reflect a point in the line.

        :Parameters:
            `point` : vec2
                The point to reflect in the line.

        """
        if not isinstance(point, vec2):
            point = vec2(point)
        offset_distance = point.dot(self.direction) - self.distance
        return point - 2 * self.direction * offset_distance

    mirror = reflect

    def distance_to(self, point):
        """Return the (signed) distance to a point.

        :Parameters:
            `point` : vec2
                The point to measure the distance to.

        """
        if not isinstance(point, vec2):
            point = vec2(point)
        return point.dot(self.direction) - self.distance

    altitude = distance_to

    def is_on_left(self, point):
        """Determine if the given point is left of the line.

        :Parameters:
            `point` : vec2
                The point to locate.

        """
        return self.distance_to(point) < 0

    is_inside = is_on_left

    def is_on_right(self, point):
        """Determine if the given point is right of the line.

        :Parameters:
            `point` : vec2
                The point to locate.

        """
        return self.distance_to(point) > 0

    def parallel(self, point):
        """Return a line parallel to this one through the given point.

        :Parameters:
            `point` : vec2
                The point through which to trace a line.

        """
        if not isinstance(point, vec2):
            point = vec2(point)
        distance = point.dot(self.direction)
        return Line(self.direction, distance)

    def perpendicular(self, point):
        """Return a line perpendicular to this one through the given point. The
        orientation of the line is consistent with ``vec2.perpendicular``.

        :Parameters:
            `point` : vec2
                The point through which to trace a line.

        """
        if not isinstance(point, vec2):
            point = vec2(point)
        direction = self.direction.perpendicular()
        distance = point.dot(direction)
        return Line(direction, distance)



class LineSegment(object):
    """Two-dimensional vector (directed) line segment implementation.

    Line segments are defined in terms of a line and the minimum and maximum
    distances from the base of the altitude to that line from the origin. The
    distances are signed, strictly they are multiples of a vector parallel to
    the line.

    """

    def __init__(self, line, min_dist, max_dist):
        """Create a LineSegment object.

        Distances are measured according to the direction of the 'along'
        attribute of the line.

        :Parameters:
            `line` : Line
                The line to take a segment of.
            `min_dist` : float
                The minimum distance from the projection of the origin.
            `max_dist` : float
                The maximum distance from the projection of the origin.

        """
        self.line = line
        self.min_dist = min_dist
        self.max_dist = max_dist

    def __str__(self):
        """Construct a concise string representation.

        """
        params = (self.line, self.min_dist, self.max_dist)
        return "LineSegment(%s, %.2f, %.2f)" % params

    def __repr__(self):
        """Construct a precise string representation.

        """
        params = (self.line, self.min_dist, self.max_dist)
        return "LineSegment(%r, %r, %r)" % params

    @classmethod
    def from_points(cls, first, second):
        """Create a LineSegment object from two (distinct) points.

        :Parameters:
            `first`, `second` : vec2
                The vectors used to construct the line.

        """
        if not isinstance(first, vec2):
            first = vec2(first)
        if not isinstance(second, vec2):
            second = vec2(second)
        line = Line.from_points(first, second)
        d1, d2 = first.dot(line.along), second.dot(line.along)
        return cls(line, min(d1, d2), max(d1, d2))

    def length(self):
        """The length of the line segment.

        """
        return abs(self.max_dist - self.min_dist)

    def _endpoints(self):
        """Compute the two endpoints of the line segment.

        """
        start = self.line.along * self.min_dist + self.line.offset
        end = self.line.along * self.max_dist + self.line.offset
        return start, end

    def start(self):
        """One endpoint of the line segment (corresponding to 'min_dist').

        """
        start, end = self._endpoints()
        self.mid = (start + end) / 2
        self.end = end
        return start

    def mid(self):
        """The midpoint of the line segment.

        """
        start, end = self._endpoints()
        self.start = start
        self.end = end
        return (start + end) / 2

    def end(self):
        """One endpoint of the line segment (corresponding to 'max_dist').

        """
        start, end = self._endpoints()
        self.start = start
        self.mid = (start + end) / 2
        return end

    def project(self, point):
        """Compute the projection of a point onto the line segment.

        :Parameters:
            `point` : vec2
                The point to minimise the distance to.

        """
        if not isinstance(point, vec2):
            point = vec2(point)
        distance = point.dot(self.line.along)
        if distance >= self.max_dist:
            return self.end
        elif distance <= self.min_dist:
            return self.start
        return self.line.along * distance + self.line.offset

    def distance_to(self, point):
        """Return the shortest distance to a given point.

        :Parameters:
            `point` : vec2
                The point to measure the distance to.

        """
        if not isinstance(point, vec2):
            point = vec2(point)
        distance = point.dot(self.line.along)
        if distance >= self.max_dist:
            return self.end.distance_to(point)
        elif distance <= self.min_dist:
            return self.start.distance_to(point)
        else:
            return abs(self.line.distance_to(point))


class Projection(object):
    """A wrapper for the extent of a projection onto a line."""

    __slots__ = 'min', 'max'
    def __init__(self, min, max):
        self.min, self.max = min, max

    def intersection(self, other):
        if self.max > other.min and other.max > self.min:
            return self.max-other.min
        return 0


class Segment(object):
    """A 2D line segment between two points p1 and p2.

    A segment has an implied direction for some operations - p1 is the start
    and p2 is the end.

    """
    def __init__(self, p1, p2):
        self.points = p1, p2
        self.edge = (p2 - p1).normalized()
        self.axis = self.edge.perpendicular()
        self.axis_dist = p1.dot(self.axis)
        self.proj = self.project_to_axis(self.edge)

    @property
    def length(self):
        """The length of the segment."""
        return abs(self.proj.max - self.proj.min)

    def scale_to(self, dist):
        """Scale the segment to be of length dist.

        This returns a new segment of length dist that shares p1 and the
        direction vector.

        """
        p1 = self.points[0]
        return Segment(p1, p1 + self.edge * dist)

    truncate = scale_to

    def project_to_axis(self, axis):
        projected_points = [p.dot(axis) for p in self.points]
        return Projection(min(projected_points), max(projected_points))

    def intersects(self, other):
        """Determine if this segment intersects a convex polygon.

        Returns None if there is no intersection, or a scalar which is how far
        along the segment the intersection starts. If the scalar is positive
        then the intersection is partway from p1 to p2. If the scalar is
        negative then p1 is inside the shape, by the corresponding distance (in
        the direction of the object)

        """
        proj = other.project_to_axis(self.axis)
        if proj.max > self.axis_dist >= proj.min:
            proj2 = other.project_to_axis(self.edge)
            if proj2.intersection(self.proj):
                return proj2.min - self.proj.min


class PolyLine(object):
    """A set of points connected into line"""

    def __init__(self, vertices=[]):
        self.vertices = vertices

    def __repr__(self):
        return 'PolyLine(%r)' % (self.vertices,)

    def __iter__(self):
        """Iterate over the vertices"""
        return iter(self.vertices)

    def segments(self):
        nvs = len(self.vertices)
        for i in range(1, nvs):
            yield LineSegment(self.vertices[i - 1], self.vertices[i])


#: The x-axis line.
x_axis = Line(unit_y, 0.0)

#: The y-axis line.
y_axis = Line(-unit_x, 0.0)
from collections import namedtuple


class BasePolygon(object):
    """Base polygon.

    Operations defined on a set of points.
    """

    def get_aabb(self):
        """Return the axis-aligned bounding box for this polygon."""
        xs, ys = zip(*self.points)
        l = min(xs)
        r = max(xs)
        b = min(ys)
        t = max(ys)
        return Rect(l, r, b, t)

    def translate(self, v):
        """Return a translated instance of this polygon."""
        return self.__class__(
            *(p + v for p in self)
        )

    def __iter__(self):
        return iter(self.points)

    def project_to_axis(self, axis):
        """Project the polygon onto the vector axis, which must be normalized.
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
            axis = edge.normalized().perpendicular()

            self_projection = self.project_to_axis(axis)
            other_projection = other.project_to_axis(axis)
            intersection1 = self_projection.intersection(other_projection)
            intersection2 = -other_projection.intersection(self_projection)
            if not intersection1:
                return False

            proj_vector1 = vec2((axis.x * intersection1, axis.y * intersection1))
            proj_vector2 = vec2((axis.x * intersection2, axis.y * intersection2))
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
            self.points.append(vec2(p))

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
            `base` : vec2
                A point vector of a base point of the triangle.
            `primary` : vec2
                The vector from the base point to one of the others.
            `secondary` : vec2
                The vector from the base point to the final point.

        """
        if not isinstance(base, vec2):
            base = vec2(base)
        if not isinstance(primary, vec2):
            primary = vec2(primary)
        if not isinstance(secondary, vec2):
            secondary = vec2(secondary)
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
            `base` : vec2
                The base point of the triangle.
            `first`, `second` : vec2
                The other two points of the triangle.

        """
        if not isinstance(base, vec2):
            base = vec2(base)
        primary = first - base
        secondary = second - base
        return cls(base, primary, secondary)

    def area(self):
        """The unsigned area of the triangle.

        """
        area = self.primary.cross(self.secondary) / 2
        self.is_clockwise = (area < 0)
        return abs(area)

    def is_clockwise(self):
        """True if the primary and secondary are clockwise.

        """
        area = self.primary.cross(self.secondary) / 2
        self.area = abs(area)
        return (area < 0)

    def first(self):
        """The point at the end of the primary vector.

        """
        return self.base + self.primary

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
        l, r = min(xs), max(xs)
        b, t = min(ys), max(ys)
        return cls(l, r, b, t)

    @property
    def points(self):
        """A list of the points in the rectangle."""
        return [
            vec2((self.l, self.b)),
            vec2((self.l, self.t)),
            vec2((self.r, self.t)),
            vec2((self.r, self.b)),
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

    def get_aabb(self):
        """Return the axis-aligned bounding box of the Rect - ie. self."""
        return self

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
        return newvec2(self.l, self.b)

    def topleft(self):
        """The top left point."""
        return newvec2(self.l, self.t)

    def topright(self):
        """The top right point."""
        return newvec2(self.r, self.t)

    def bottomright(self):
        """The bottom right point."""
        return newvec2(self.r, self.b)

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



class SpatialHash:
    def __init__(self, cell_size=250.0):
        self.cell_size = float(cell_size)
        self.d = {}

    def _add(self, cell_coord, o):
        """Add the object o to the cell at cell_coord."""
        self.d.setdefault(cell_coord, set()).add(o)

    def _cells_for_rect(self, r: Rect):
        """Return a set of the cells into which r extends."""
        cells = set()
        cy = floor(r.b / self.cell_size)
        while (cy * self.cell_size) <= r.t:
            cx = floor(r.l / self.cell_size)
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


cdef class Matrix:
    """A 2x2 matrix.

    This can be used to optimise a transform (such as a rotation) on multiple
    vectors without recomputing terms.

    To transform a vector with this matrix, use premultiplication, ie. for
    Matrix M and vec2 v, ::

        t = M * v

    """
    cdef double x11, x12, x21, x22

    @staticmethod
    cdef new(double x11, double x12, double x21, double x22):
        cdef Matrix m
        m = Matrix.__new__(Matrix)
        m.x11 = x11
        m.x12 = x12
        m.x21 = x21
        m.x22 = x22
        return m

    def __init__(self, double x11, double x12, double x21, double x22):
        self.x11 = x11
        self.x12 = x12
        self.x21 = x21
        self.x22 = x22

    def __mul__(Matrix self, vec2 vec):
        """Multiple a vector by this matrix."""
        return newvec2(
            self.x11 * vec.x + self.x12 * vec.y,
            self.x21 * vec.x + self.x22 * vec.y
        )

    @staticmethod
    def identity():
        return Matrix.new(1.0, 0.0, 0.0, 1.0)

    @staticmethod
    def rotation(double angle):
        """A rotation matrix for angle a."""
        cdef double s, c
        s = sin(angle)
        c = cos(angle)
        return Matrix.new(c, -s, s, c)


def bresenham(int64_t x0, int64_t y0, int64_t x1, int64_t y1):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).

    Input coordinates should be integers.
    The result will contain both the start and the end point.

    """
    # Copyright © 2016 Petr Viktorin
    # Licensed under MIT
    # Copied from https://github.com/encukou/bresenham and cythonised

    cdef int64_t dx, dy, xx, xy, yx, yy, D, y, x
    cdef bint xsign, ysign

    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = llabs(dx)
    dy = llabs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy
