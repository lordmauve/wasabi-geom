#cython: language_level=3, c_api_binop_methods=True

cimport cython
from libc.math cimport sqrt, atan2, cos, sin, floor, pi, asin, acos, ceil
from libc.stdlib cimport llabs
from libc.stdint cimport int64_t, uint64_t
from cpython.sequence cimport PySequence_Check
from cpython.unicode cimport PyUnicode_AsUTF8
from cpython.exc cimport PyErr_WarnEx
from cpython cimport Py_buffer
from cython cimport floating


cdef inline int _extract(object o, double *x, double *y) except -1:
    cdef int64_t l
    cdef double tmp
    if isinstance(o, vec2):
        x[0] = (<vec2> o).x
        y[0] = (<vec2> o).y
        return 1

    if not PySequence_Check(o):
        return 0
    if len(o) != 2:
        raise TypeError("Tuple was not of length 2")

    tmp = <double?> o[0]
    y[0] = <double?> o[1]
    x[0] = tmp
    return 1


cdef vec2 newvec2(double x, double y):
    cdef vec2 v = vec2.__new__(vec2)
    v.x = x
    v.y = y
    return v


cdef str RADIANS_WARNING = """\
You called a method vec2.{}() which uses angles in radians, not angles in \
degrees. Radians are the "natural" unit of angle in mathematics and computing. \
A radian is about 57.3 degrees. 180 degrees is pi radians (about 3.14).

If you want to use degrees, use the function vec2.{}(). To explicitly use \
radians, use vec2.{}().

If you are happy using radians, call vec2.radians_are_awesome() to suppress \
this message globally.\
"""


cdef bint radians_unsure = 0


class RadiansWarning(Warning):
    """A warning about the use of radians.""""


cdef inline void warn_radians(func: str, deg_func: str, rad_func: str) noexcept:
    """Warn about the use of radians."""
    global radians_unsure
    if radians_unsure:
        warning = RADIANS_WARNING.format(func, deg_func, rad_func)
        PyErr_WarnEx(RuntimeWarning, warning.encode(), 1)
        radians_unsure = 0


cdef double degrees(double radians) noexcept:
    return radians * (180.0 / pi)


cdef double radians(double degrees) noexcept:
    return degrees * (pi / 180.0)


@cython.freelist(32)
@cython.final
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

    @staticmethod
    def radians_are_awesome(v: bool = True) -> None:
        global radians_unsure
        radians_unsure = not v

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
        warn_radians("angle", "angle_degrees", "angle_radians")
        return self.angle_radians()

    cpdef double angle_radians(self):
        """The angle the vector makes to the positive x axis, in radians."""
        return atan2(self.y, self.x)

    cpdef double angle_degrees(self):
        """The angle the vector makes to the positive x axis, in degrees."""
        return degrees(atan2(self.y, self.x))

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

    def __mul__(object a, object b):
        """Multiply the vector by a scalar.

        Use .dot() to compute the dot product.

        :Parameters:
            `other` : float
                The object by which to multiply.

        """
        cdef double f
        cdef vec2 vec
        cdef object other
        if isinstance(a, vec2):
            vec = <vec2> a
            other = b
        else:
            vec = <vec2> b
            other = a
        f = <double?> other
        return newvec2(vec.x * f, vec.y * f)

    def __truediv__(a, b):
        """Divide the vector by a scalar."""
        cdef double f, x, y
        if isinstance(a, vec2):
            if not isinstance(b, (int, float)):
                return NotImplemented
            x = (<vec2> a).x
            y = (<vec2> a).y
            f = <double ?> b
            return newvec2(x / f, y / f)
        else:
            if not isinstance(a, (int, float)):
                return NotImplemented
            x = (<vec2> b).x
            y = (<vec2> b).y
            f = <double ?> a
            return newvec2(f / x, f / y)

    def __floordiv__(a, b):
        """Divide the vector by a scalar, rounding down."""
        cdef double f, x, y
        if isinstance(a, vec2):
            if not isinstance(b, (int, float)):
                return NotImplemented
            x = (<vec2> a).x
            y = (<vec2> a).y
            f = <double ?> b
            return newvec2(x // f, y // f)
        else:
            if not isinstance(a, (int, float)):
                return NotImplemented
            x = (<vec2> b).x
            y = (<vec2> b).y
            f = <double ?> a
            return newvec2(f // x, f // y)

    def __neg__(vec2 self):
        """Compute the unary negation of the vector."""
        return newvec2(-self.x, -self.y)

    cpdef vec2 rotated_degrees(vec2 self, double angle):
        """Compute the vector rotated by an angle in degrees.

        :Parameters:
            `angle` : float
                The angle (in degrees) by which to rotate.

        """
        return self.rotated_radians(radians(angle))

    cpdef vec2 rotated(vec2 self, double angle):
        """Compute the vector rotated by an angle in radians, with a warning.

        :Parameters:
            `angle` : float
                The angle (in radians) by which to rotate.

        """
        warn_radians("rotated", "rotated_degrees", "rotated_radians")
        return self.rotated_radians(angle)

    cpdef vec2 rotated_radians(vec2 self, double angle):
        """Compute the vector rotated by an angle.

        :Parameters:
            `angle` : float
                The angle (in radians) by which to rotate.

        """
        cdef double vx, vy, ca, sa
        vx = self.x
        vy = self.y
        ca = cos(angle)
        sa = sin(angle)
        return newvec2(vx * ca - vy * sa, vx * sa + vy * ca)

    def scaled_to(vec2 self, double length):
        """Compute the vector scaled to a given length.

        :Parameters:
            `length` : float
                The length to which to scale.

        """
        cdef double vx, vy, s
        vx = self.x
        vy = self.y
        s = length / self.length()
        return newvec2(vx * s, vy * s)

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

    cpdef normalized(self):
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

    def perpendicular(vec2 self):
        """Compute the perpendicular."""
        return newvec2(-self.y, self.x)

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

    def cross(vec2 self, other):
        """Compute the cross product with another vector.

        :Parameters:
            `other` : vec2
                The vector with which to compute the cross product.

        """
        cdef double x2, y2
        if not _extract(other, &x2, &y2):
            raise TypeError("Expected vec2 or 2-tuple")
        return self.x * y2 - self.y * x2

    def project(self, other):
        """Compute the projection of another vector onto this one.

        :Parameters:
            `other` : vec2
                The vector of which to compute the projection.

        """
        return self * self.dot(other) / self.dot(self)

    cpdef double angle_to_radians(self, other):
        """Compute the angle made to another vector in the range [0, pi].

        :Parameters:
            `other` : vec2
                The vector with which to compute the angle.

        """
        if not isinstance(other, vec2):
            other = vec2(other)
        a = abs(other.angle_radians() - self.angle_radians())
        return min(a, 2 * pi - a)

    cpdef double angle_to_degrees(self, other):
        """Compute the angle made to another vector in degrees in the range [0, 180].

        :Parameters:
            `other` : vec2
                The vector with which to compute the angle.

        """
        return degrees(self.angle_to_radians(other))

    cpdef double angle_to(self, other):
        """Compute the angle made to another vector in radians in the range [0, pi].

        :Parameters:
            `other` : vec2
                The vector with which to compute the angle.

        """
        warn_radians("angle_to", "angle_to_degrees", "angle_to_radians")
        return self.angle_to_radians(other)

    cpdef double signed_angle_to_radians(self, other):
        """Compute the signed angle made to another vector in the range.

        :Parameters:
            `other` : vec2
                The vector with which to compute the angle.

        """
        if not isinstance(other, vec2):
            other = vec2(other)
        a = other.angle() - self.angle()
        return min(a + pi, a, a - pi, key=abs)

    cpdef double signed_angle_to_degrees(self, other):
        """Compute the signed angle made to another vector in degrees.

        :Parameters:
            `other` : vec2
                The vector with which to compute the angle.

        """
        return degrees(self.signed_angle_to_radians(other))

    cpdef double signed_angle_to(self, other):
        """Compute the signed angle made to another vector in radians.

        :Parameters:
            `other` : vec2
                The vector with which to compute the angle.

        """
        warn_radians("signed_angle_to", "signed_angle_to_degrees", "signed_angle_to_radians")
        return self.signed_angle_to_radians(other)

    cpdef (double, double) to_polar_radians(vec2 self):
        """Return (length, angle) in polar coordinates, with angle in radians."""
        return self.length(), self.angle_radians()

    cpdef (double, double) to_polar_degrees(vec2 self):
        """Return (length, angle) in polar coordinates, with angle in degrees."""
        return self.length(), self.angle_degrees()

    cpdef (double, double) to_polar(vec2 self):
        """Return (length, angle) in polar coordinates, with angle in radians, and a warning."""
        warn_radians("to_polar", "to_polar_degrees", "to_polar_radians")
        return self.length(), self.angle_radians()

    @staticmethod
    def from_polar_radians(double length, double angle):
        """Construct a vec2 from polar coordinates with angle in radians."""
        return newvec2(length * cos(angle), length * sin(angle))

    @staticmethod
    def from_polar_degrees(double length, double angle):
        """Construct a vec2 from polar coordinates with angle in degrees."""
        angle = radians(angle)
        return newvec2(length * cos(angle), length * sin(angle))

    @staticmethod
    def from_polar(double length, double angle):
        """Construct a vec2 from polar coordinates with angle in radians, with a warning."""
        warn_radians("from_polar", "from_polar_degrees", "from_polar_radians")
        return newvec2(length * cos(angle), length * sin(angle))

    def distance_to(vec2 self, other):
        """Compute the distance to another point vector.

        :Parameters:
            `other` : vec2
                The point vector to which to compute the distance.

        """
        cdef double x2, y2
        if not _extract(other, &x2, &y2):
            raise TypeError("Expected vec2 or 2-tuple")
        x2 -= self.x
        y2 -= self.y
        return sqrt(x2 * x2 + y2 * y2)


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


@cython.freelist(32)
cdef class Transform:
    """A 3x3 matrix representing an affine transform in 2D.

    The values of the matrix are ::

        (a b c)
        (d e f)
        (0 0 1)

    These matrices always a bottom row (0, 0, 1), and knowing this allows us to
    skip some multiplications and drop some terms vs multiplication of an
    arbitrary 3x3 matrix (eg. in numpy).

    Transforms can be multiplied together to chain them::

        >>> a = Transform.build(xlate=(1, 2), rot=0.5)
        >>> b = Transform.build(rot=-0.5)
        >>> a * b
        Transform(1., 0., 1.,
                  0., 1., 2.)

    Order matters! Matrix multiplication is not commutative (but it is
    associative). To do transformation ``A`` followed by ``B`` is ``B * A``.

    Transforms support the buffer protocol, meaning that they can be converted
    to numpy arrays::

        >>> numpy.asarray(Transform(2., 0., 1.,
        ...                         0., 1., 2.))
        array([[2., 0., 1.],
               [0., 1., 2.]])

    """
    cdef double a, b, c, d, e, f

    @staticmethod
    def identity():
        """Return a new identity transform."""
        cdef Transform m
        m = Transform.__new__(Transform)
        m.a = 1.0
        m.b = 0.0
        m.c = 0.0

        m.d = 0.0
        m.e = 1.0
        m.f = 0.0
        return m

    @staticmethod
    def build(object xlate = (0, 0), double rot = 0.0, object scale = (1, 1)):
        """Build a Transform from a translation, rotation and scale.

        The operation order is scale first, then rotation, then translation. To
        apply the operations different order, build Transforms representing the
        individual operations and then multiply them.

        """
        cdef Transform m
        m = Transform.__new__(Transform)
        m.set(xlate, rot, scale)
        return m

    def set(self, object xlate = (0, 0), double rot = 0.0, object scale = (1, 1)):
        """Overwrite the transform using the given parameters."""
        cdef double tx, ty
        cdef double sx, sy
        cdef double cos_theta, sin_theta

        if not _extract(xlate, &tx, &ty) or not _extract(scale, &sx, &sy):
            raise TypeError("xlate and scale must be 2d vectors")

        cos_theta = cos(rot)
        sin_theta = sin(rot)

        self.a = sx * cos_theta
        self.b = sy * -sin_theta
        self.c = tx

        self.d = sx * sin_theta
        self.e = sy * cos_theta
        self.f = ty

    def __init__(self, double a, double b, double c, double d, double e, double f):
        """Construct a Transform matrix from its components."""
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def __mul__(object obja, object objb):
        """Multiply the transform by another transform."""
        if isinstance(obja, Transform) and isinstance(objb, Transform):
            return Transform.matmul(obja, objb)

        return NotImplemented

    @staticmethod
    cdef matmul(Transform A, Transform B):
        cdef Transform m
        m = Transform.__new__(Transform)
        m.a = A.a * B.a + A.b * B.d
        m.b = A.a * B.b + A.b * B.e
        m.c = A.a * B.c + A.b * B.f + A.c

        m.d = A.d * B.a + A.e * B.d
        m.e = A.d * B.b + A.e * B.e
        m.f = A.d * B.c + A.e * B.f + A.f

        return m

    def inverse(self):
        """Return the inverse transformation.

        Raise ZeroDivisionError if the matrix is not invertible (eg. it has a
        scale factor of 0).

        """
        cdef Transform m
        cdef double det

        det = self.a * self.e - self.b * self.d
        if det == 0.0:
            raise ZeroDivisionError("Non-invertable matrix")

        m = Transform.__new__(Transform)
        m.a = self.e / det
        m.b = -self.b / det
        m.c = (self.b * self.f - self.c * self.e) / det

        m.d = -self.d / det
        m.e = self.a / det
        m.f = (-self.a * self.f + self.c * self.d) / det

        return m

    def factorise(self):
        """Split the transformation into translation, rotation, and scale.

        This operation is approximate because it doesn't calculate or return
        skew components, and therefore cannot represent all transforms.

        """
        cdef double scale_x, scale_y, angle
        scale_x = sqrt(self.a * self.a + self.b * self.b)
        scale_y = sqrt(self.d * self.d + self.e * self.e)

        angle = atan2(self.b, self.a)

        return (newvec2(self.c, self.f), angle, newvec2(scale_x, scale_y))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def transform(self,
                  floating[:,:] input_view not None,
                  floating[:,:] output_view=None):
        """Transform a buffer of coordinates using this matrix.

        Coordinates may be floats or doubles.

        If output_view is given, then it will be populated with the result
        rather than returning a new numpy arry. It should be a writable buffer
        object matching the shape of the input.

        :param input_view: A 2D array of 2 coordinates to transform.
        :param output_view: A 2D array of 2 coordinates to write to, or None
                            to allocate a new array.
        :returns: A numpy.ndarray, unless output_view is given.


        """
        cdef floating x, y

        ret = None

        if output_view is None:
            import numpy as np
            # Creating a default view, e.g.
            ret = np.empty_like(input_view)
            output_view = ret

        if input_view.shape[0] != output_view.shape[0]:
            raise TypeError("Length of input view must match output")

        if not (input_view.shape[1] == output_view.shape[1] == 2):
            raise TypeError("Can only transform arrays of 2D vectors")

        with nogil:
            for i in range(input_view.shape[0]):
                x = input_view[i, 0]
                y = input_view[i, 1]
                output_view[i, 0] = self.a * x + self.b * y + self.c
                output_view[i, 1] = self.d * x + self.e * y + self.f
        return ret

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(self.a)

        buffer.buf = &self.a
        buffer.format = 'd'                     # double
        buffer.internal = NULL                  # see References
        buffer.itemsize = itemsize
        buffer.len = 6 * itemsize   # product(shape) * itemsize
        buffer.ndim = 2
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = XFORM_SHAPE
        buffer.strides = XFORM_STRIDES
        buffer.suboffsets = NULL                # for pointer arrays only

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __repr__(self):
        return (
            f'Transform({self.a}, {self.b}, {self.c},\n'
            f'          {self.d}, {self.e}, {self.f})'
        )


cdef Py_ssize_t[2] XFORM_SHAPE = [2, 3]
cdef Py_ssize_t[2] XFORM_STRIDES = [3 * sizeof(double), sizeof(double)]


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


cdef ZRect asrect(object obj):
    if isinstance(obj, ZRect):
        return obj
    return ZRect(obj)


cdef ZRect newzrect(x: float, y: float, w: float, h: float):
    r = ZRect.__new__(ZRect)
    r.x = x
    r.y = y
    r.w = w
    r.h = h
    return r


cdef class ZRect:
    """ZRect

    This is a Cython implementation of the pygame Rect class. Its raison
    d'être is to allow the coordinates to be floating point. All pygame
    functions which require a rect allow for an object with a "rect"
    attribute and whose coordinates will be converted to integers implictly.

    All functions which require a dict will use the flexible constructor
    to convert from: this (or a subclass); a Pygame Rect; a 4-tuple or a
    pair of 2-tuples. In addition, they'll recognise any object which has
    an (optionally callable) .rect attribute whose value will be used instead.
    """
    cdef public double x, y, w, h

    def __init__(self, *args):

        if len(args) == 1:
            args = tuple(self._handle_one_arg(args[0]))

        #
        # At this point we have one of:
        #
        # x, y, w, h
        # (x, y), (w, h)
        # (x, y, w, h),
        #
        if len(args) == 4:
            self.x, self.y, self.w, self.h = args
        elif len(args) == 2:
            _extract(args[0], &self.x, &self.y)
            _extract(args[1], &self.w, &self.h)
        elif len(args) == 1:
            self.x, self.y, self.w, self.h = args[0]
        else:
            raise TypeError(
                "%s should be called with one, two or four arguments"
                % (self.__class__.__name__)
            )

    cdef _handle_one_arg(self, arg):
        """Handle -- possibly recursively -- the case of one parameter

        Pygame -- and consequently pgzero -- is very accommodating when constructing
        a rect. You can pass four integers, two pairs of 2-tuples, or one 4-tuple.

        Also, you can pass an existing Rect-like object, or an object with a .rect
        attribute. The object named by the .rect attribute is either one of the above,
        or it's a callable object which returns one of the above.

        This is evidently a recursive solution where an object with a .rect
        attribute can yield an object with a .rect attribute, and so ad infinitum.
        """
        #
        # If it's something with a .rect attribute, start again with
        # that attribute, calling it first if it's callable
        #
        if hasattr(arg, "rect"):
            rectobj = arg.rect
            if callable(rectobj):
                rectobj = rectobj()
            return self._handle_one_arg(rectobj)

        #
        # Otherwise, we assume it's an iterable of four elements
        #
        return arg

    def __repr__(self):
        return "<%s (x: %s, y: %s, w: %s, h: %s)>" % (
            self.__class__.__name__, self.x, self.y, self.w, self.h)

    def __reduce__(self):
        return self.__class__, (self.x, self.y, self.w, self.h)

    def copy(self):
        return newzrect(self.x, self.y, self.w, self.h)

    __copy__ = copy

    def __len__(self):
        return 4

    def __getitem__(self, item: cython.int) -> float:
        if 0 <= item < 4:
            return (&self.x)[item]
        raise IndexError

    def __setitem__(self, item: cython.int, value: float) -> None:
        if 0 <= item < 4:
            (&self.x)[item] = value
        raise IndexError

    def __bool__(self) -> bool:
        return self.w != 0 and self.h != 0

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.w
        yield self.h

    def __hash__(self):
        raise TypeError("ZRect instances may not be used as dictionary keys")

    def __eq__(self, other):
        cdef ZRect rect = asrect(other)
        cdef bint eq = (
            self.x == rect.x
            and self.y == rect.y
            and self.w == rect.w
            and self.h == rect.h
        )
        return eq

    def __ne__(self, other):
        cdef ZRect rect = asrect(other)
        cdef bint ne = (
            self.x != rect.x
            or self.y != rect.y
            or self.w != rect.w
            or self.h != rect.h
        )
        return ne

    def __contains__(self, other):
        """Test whether a point (x, y) or another rectangle
        (anything accepted by ZRect) is contained within this ZRect
        """
        if len(other) == 2:
            return self.collidepoint(*other)
        else:
            return self.contains(*other)

    @property
    def width(self) -> float:
        return self.w

    @width.setter
    def width(self, width: cython.double) -> None:
        self.w = width

    @property
    def height(self) -> float:
        return self.h

    @height.setter
    def height(self, height: float) -> None:
        self.h = height

    @property
    def top(self) -> float:
        return self.y

    @top.setter
    def top(self, top: float) -> None:
        self.y = top

    @property
    def left(self) -> float:
        return self.x

    @left.setter
    def left(self, left: float) -> None:
        self.x = left

    @property
    def right(self) -> float:
        return self.x + self.w

    @right.setter
    def right(self, right: float) -> None:
        self.x = right - self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @bottom.setter
    def bottom(self, bottom: float) -> None:
        self.y = bottom - self.h

    @property
    def centerx(self) -> float:
        return self.x + (self.w / 2)

    @centerx.setter
    def centerx(self, centerx):
        self.x = centerx - (self.w / 2)

    @property
    def centery(self) -> float:
        return self.y + (self.h / 2)

    @centery.setter
    def centery(self, centery: float) -> None:
        self.y = centery - (self.h / 2)

    @property
    def topleft(self) -> vec2:
        return newvec2(self.x, self.y)

    @topleft.setter
    def topleft(self, topleft):
        _extract(topleft, &self.x, &self.y)

    @property
    def topright(self) -> float:
        return self.x + self.w, self.y

    @topright.setter
    def topright(self, topright: object) -> None:
        cdef double right
        _extract(topright, &right, &self.y)
        self.x = right - self.w

    @property
    def bottomleft(self):
        return newvec2(self.x, self.y + self.h)

    @bottomleft.setter
    def bottomleft(self, bottomleft):
        cdef double bottom
        _extract(bottomleft, &self.x, &bottom)
        self.y = bottom - self.h

    @property
    def bottomright(self):
        return newvec2(self.x + self.w, self.y + self.h)

    @bottomright.setter
    def bottomright(self, bottomright):
        cdef double right, bottom
        _extract(bottomright, &right, &bottom)
        self.x = right - self.w
        self.y = bottom - self.h

    @property
    def midtop(self) -> vec2:
        return newvec2(self.x + self.w / 2, self.y)

    @midtop.setter
    def midtop(self, midtop):
        cdef double x
        _extract(midtop, &x, &self.y)
        self.x = x - self.w / 2

    @property
    def midleft(self):
        return newvec2(self.x, self.y + self.h / 2)

    @midleft.setter
    def midleft(self, midleft):
        cdef double y
        _extract(midleft, &self.x, &y)
        self.y = y - self.h / 2

    @property
    def midbottom(self):
        return newvec2(self.x + self.w / 2, self.y + self.h)

    @midbottom.setter
    def midbottom(self, midbottom):
        cdef double x, y
        _extract(midbottom, &x, &y)
        self.x = x - self.w / 2
        self.y = y - self.h

    @property
    def midright(self):
        return newvec2(self.x + self.w, self.y + self.h / 2)

    @midright.setter
    def midright(self, midright):
        cdef double x, y
        _extract(midright, &x, &y)
        self.x = x - self.w
        self.y = y - self.h / 2

    @property
    def center(self):
        return newvec2(self.x + self.w / 2, self.y + self.h / 2)

    @center.setter
    def center(self, center):
        cdef double x, y
        _extract(center, &x, &y)
        self.x = x - self.w / 2
        self.y = y - self.h / 2

    @property
    def size(self):
        return newvec2(self.w, self.h)

    @size.setter
    def size(self, size):
        _extract(size, &self.w, &self.h)

    def move(self, x: float, y: float) -> ZRect:
        return newzrect(self.x + x, self.y + y, self.w, self.h)

    def move_ip(self, x: float, y: float):
        self.x += x
        self.y += y

    cdef _inflated(
        ZRect self,
        double x,
        double y,
        ZRect out,
    ):
        out.x = self.x - x / 2
        out.y = self.y - y / 2
        out.w = self.w + x
        out.h = self.h + y

    def inflate(self, x: float, y: float) -> ZRect:
        cdef ZRect r = ZRect.__new__(ZRect)
        self._inflated(x, y, r)
        return r

    def inflate_ip(self: ZRect, x: float, y: float) -> None:
        self._inflated(x, y, self)

    cdef (double, double) _clamped(self, ZRect rect):
        if self.w >= rect.w:
            x = rect.x + rect.w / 2 - self.w / 2
        elif self.x < rect.x:
            x = rect.x
        elif self.x + self.w > rect.x + rect.w:
            x = rect.x + rect.w - self.w
        else:
            x = self.x

        if self.h >= rect.h:
            y = rect.y + rect.h / 2 - self.h / 2
        elif self.y < rect.y:
            y = rect.y
        elif self.y + self.h > rect.y + rect.h:
            y = rect.y + rect.h - self.h
        else:
            y = self.y

        return x, y

    def clamp(self, *other):
        cdef double x, y
        x, y = self._clamped(ZRect(*other))
        return newzrect(x, y, self.w, self.h)

    def clamp_ip(self, *other):
        self.x, self.y = self._clamped(ZRect(*other))

    cdef (double, double, double, double) _clipped(self, ZRect rect):
        cdef double x, y, w, h

        cdef bint intersect = True
        if self.x >= rect.x and self.x < (rect.x + rect.w):
            x = self.x
        elif rect.x >= self.x and rect.x < (self.x + self.w):
            x = rect.x
        else:
            intersect = False

        if (self.x + self.w) > rect.x and (self.x + self.w) <= (rect.x + rect.w):
            w = self.x + self.w - x
        elif (rect.x + rect.w) > self.x and (rect.x + rect.w) <= (self.x + self.w):
            w = rect.x + rect.w - x
        else:
            intersect = False

        if self.y >= rect.y and self.y < (rect.y + rect.h):
            y = self.y
        elif rect.y >= self.y and rect.y < (self.y + self.h):
            y = rect.y
        else:
            intersect = False

        if (self.y + self.h) > rect.y and (self.y + self.h) <= (rect.y + rect.h):
            h = self.y + self.h - y
        elif (rect.y + rect.h) > self.y and (rect.y + rect.h) <= (self.y + self.h):
            h = rect.y + rect.h - y
        else:
            intersect = False

        if intersect:
            return x, y, w, h
        return self.x, self.y, 0, 0

    def clip(self, *other):
        cdef double x, y, w, h
        x, y, w, h = self._clipped(ZRect(*other))
        return newzrect(x, y, w, h)

    def clip_ip(self, *other):
        self.x, self.y, self.w, self.h = self._clipped(ZRect(*other))

    cdef void _union(ZRect out, ZRect rect):
        cdef double x, y
        x = min(out.x, rect.x)
        y = min(out.y, rect.y)
        out.w = max(out.x + out.w, rect.x + rect.w) - x
        out.h = max(out.y + out.h, rect.y + rect.h) - y
        out.x = x
        out.y = y

    def union(self, *other):
        cdef ZRect out = newzrect(self.x, self.y, self.w, self.h)
        ZRect._union(out, ZRect(*other))
        return out

    def union_ip(self, *other):
        ZRect._union(self, ZRect(*other))

    def unionall(self, others):
        cdef ZRect out = newzrect(self.x, self.y, self.w, self.h)
        for r in others:
            ZRect._union(out, asrect(r))
        return out

    def unionall_ip(self, others):
        # Build a new Rect first in case any type conversion fails
        cdef ZRect union = self.unionall(others)
        self.x = union.x
        self.y = union.y
        self.w = union.w
        self.h = union.h

    def fit(self, *other):
        cdef ZRect rect = ZRect(*other)
        cdef double x, y, w, h
        ratio = max(self.w / rect.w, self.h / rect.h)
        w = self.w / ratio
        h = self.h / ratio
        x = rect.x + (rect.w - w) / 2
        y = rect.y + (rect.h - h) / 2
        return newzrect(x, y, w, h)

    def normalize(self):
        if self.w < 0:
            self.x += self.w
            self.w = abs(self.w)
        if self.h < 0:
            self.y += self.h
            self.h = abs(self.h)

    def contains(self, *other) -> bool:
        cdef ZRect rect = ZRect(*other)
        cdef bint contains = (
            self.x <= rect.x and
            self.y <= rect.y and
            self.x + self.w >= rect.x + rect.w and
            self.y + self.h >= rect.y + rect.h and
            self.x + self.w > rect.x and
            self.y + self.h > rect.y
        )
        return contains

    def collidepoint(self, *args) -> bool:
        cdef double x, y
        if len(args) == 1:
            _extract(args[0], &x, &y)
        else:
            x, y = args
        cdef bint collides = (
            self.x <= x < (self.x + self.w) and
            self.y <= y < (self.y + self.h)
        )
        return collides

    def colliderect(self, *other):
        cdef ZRect rect = ZRect(*other)
        cdef bint collides = (
            self.x < rect.x + rect.w and
            self.y < rect.y + rect.h and
            self.x + self.w > rect.x and
            self.y + self.h > rect.y
        )
        return collides

    def collidelist(self, others):
        for n, other in enumerate(others):
            if self.colliderect(other):
                return n
        else:
            return -1

    def collidelistall(self, others):
        return [n for n, other in enumerate(others) if self.colliderect(other)]

    def collidedict(self, dict: dict, use_values: bool = True):
        for k, v in dict.items():
            if self.colliderect(v if use_values else k):
                return k, v

    def collidedictall(self, dict: dict, use_values: bool = True):
        if use_values:
            return [i for i in dict.items() if self.colliderect(i[1])]
        else:
            return [i for i in dict.items() if self.colliderect(i[0])]


cdef inline void _quad_bezier_subdivide(vec2 v0, vec2 v1, vec2 v2, double tol, list acc):
    """
    Recursively subdivide the quadratic Bézier curve defined by control points v0, v1, v2,
    appending the start point of each flat segment into the accumulator list 'acc'.
    The subdivision is based on the distance between the curve's midpoint and the chord's midpoint.

    Parameters:
      v0, v1, v2 : vec2
          The control points defining the Bézier segment.
      tol : double
          The tolerance (in metres) for flatness.
      acc : list
          The accumulator list where points are appended.
    """
    # Calculate the midpoints using De Casteljau's algorithm.
    cdef vec2 m0 = (v0 + v1) * 0.5
    cdef vec2 m1 = (v1 + v2) * 0.5
    cdef vec2 m  = (m0 + m1) * 0.5
    cdef vec2 chord_mid = (v0 + v2) * 0.5

    # If the distance between the curve's midpoint and the chord's midpoint is within tolerance,
    # consider this segment flat enough.
    if m.distance_to(chord_mid) < tol:
        acc.append(v0)
    else:
        # Recursively subdivide the left and right segments.
        _quad_bezier_subdivide(v0, m0, m, tol, acc)
        _quad_bezier_subdivide(m, m1, v2, tol, acc)


cdef vec2 asvec(object obj):
    cdef vec2 v
    cdef double x, y
    _extract(obj, &x, &y)
    return newvec2(x, y)


def quadratic_bezier(v0, v1, v2, double tolerance=0.5) -> list[vec2]:
    """
    Approximate a quadratic Bézier curve (defined by control points v0, v1, v2)
    with a series of points computed via adaptive subdivision.

    Parameters:
      v0, v1, v2 : vec2
          The control points. The curve starts at v0 and ends at v2. v1 is the
          control point.
      tolerance : double
          The maximum allowed deviation between the true curve midpoint and the
          chord midpoint.

    Returns:
      A list of vec2 points that approximate the curve with each segment being
      sufficiently flat.
    """
    # The algorithm works by recursively subdividing the curve until each
    # segment is flat enough. Only the start point of each flat segment is
    # stored, and the final control point is appended at the end.
    cdef list points = []

    cdef v0vec = asvec(v0)
    cdef v1vec = asvec(v1)
    cdef v2vec = asvec(v2)

    _quad_bezier_subdivide(v0vec, v1vec, v2vec, tolerance, points)
    points.append(v2vec)
    return points


cdef inline void _cubic_bezier_subdivide(vec2 v0, vec2 v1, vec2 v2, vec2 v3, double tol, list acc, unsigned int level):
    """
    Recursively subdivide a cubic Bézier curve defined by control points v0, v1, v2, v3.
    Uses the De Casteljau algorithm to compute intermediate midpoints and compares the
    true curve midpoint (p0123) to the chord midpoint ( (v0+v3)/2 ). If the distance between
    these is below tol, the segment is considered flat and v0 is appended to acc. Otherwise,
    the curve is subdivided into two halves.
    """
    # Compute first-level midpoints.
    cdef vec2 p01 = (v0 + v1) * 0.5
    cdef vec2 p12 = (v1 + v2) * 0.5
    cdef vec2 p23 = (v2 + v3) * 0.5

    # Compute second-level midpoints.
    cdef vec2 p012 = (p01 + p12) * 0.5
    cdef vec2 p123 = (p12 + p23) * 0.5

    # Compute the curve midpoint.
    cdef vec2 p0123 = (p012 + p123) * 0.5

    # Compute the chord midpoint.
    cdef vec2 chord_mid = (v0 + v3) * 0.5

    # If the deviation is within tolerance, consider the segment flat.
    if level > 0 and p0123.distance_to(chord_mid) < tol:
        acc.append(v0)
    else:
        _cubic_bezier_subdivide(v0, p01, p012, p0123, tol, acc, 1)
        _cubic_bezier_subdivide(p0123, p123, p23, v3, tol, acc, 1)


def cubic_bezier(v0, v1, v2, v3, double tolerance=0.5) -> list[vec2]:
    """
    Approximate a cubic Bézier curve (defined by control points v0, v1, v2, v3)
    with a series of points computed via adaptive subdivision.

    Parameters:
      v0, v1, v2, v3 : vec2
          The control points. v0 and v3 are the start and end points of the
          curve; v1 and v2 are the control points.
      tolerance : double
          The maximum allowed deviation between the true curve midpoint and the
          chord midpoint.

    Returns:
      A list of vec2 points that approximate the curve with each segment being
      sufficiently flat.
    """
    cdef list points = []

    cdef vec2 v0vec = asvec(v0)
    cdef vec2 v1vec = asvec(v1)
    cdef vec2 v2vec = asvec(v2)
    cdef vec2 v3vec = asvec(v3)

    _cubic_bezier_subdivide(v0vec, v1vec, v2vec, v3vec, tolerance, points, 0)
    points.append(v3vec)
    return points


def circular_arc(vec2 p0, vec2 p1, double radius, bint clockwise, double tolerance=0.5) -> list:
    """
    Compute points along a circular arc from p0 to p1 for a circle with the given radius.
    The arc is drawn in the specified direction (clockwise if True) and subdivided into
    segments such that the maximum deviation (sagitta) of each chord from the true arc is
    within the tolerance.

    Parameters:
      p0, p1 : vec2
          The end‑points of the arc.
      radius : float
          The radius of the circle. The chord between p0 and p1 must be at most 2*radius.
      clockwise : bool
          If True the arc is drawn in a clockwise direction; otherwise, anticlockwise.
      tolerance : float, default 0.5
          The maximum allowed sagitta for each segment.

    Returns:
      A list of vec2 points along the arc.
    """
    cdef double d2, d, dx, dy, h, mx, my, cx, cy, start_angle, theta, sweep, dtheta_max, dtheta, angle, x, y
    cdef vec2 sep, m, center
    cdef double perp_x, perp_y
    cdef int n_segments, i
    cdef list points = []

    # Compute the chord vector and its length (squared).
    sep = p1 - p0
    dx = p1.x - p0.x
    dy = p1.y - p0.y
    d2 = sep.length_squared()
    if d2 > 4.0 * radius * radius:
        raise ValueError("Chord length exceeds diameter for the given radius")
    if d2 == 0.0:
        return [p0]

    d = sqrt(d2)

    # Compute the chord midpoint using vec2 arithmetic.
    m = p0 + sep * 0.5
    mx = m.x
    my = m.y

    # Compute h = √(radius² - (d/2)²)
    h = sqrt(radius * radius - d2 / 4.0)

    # Determine the unit-length perpendicular vector.
    if clockwise:
        perp_x = dy / d
        perp_y = -dx / d
    else:
        perp_x = -dy / d
        perp_y = dx / d

    # Determine the circle center.
    cx = mx + h * perp_x
    cy = my + h * perp_y
    center = vec2(cx, cy)

    # Compute the starting angle (angle from center to p0).
    start_angle = atan2(p0.y - cy, p0.x - cx)
    # The chord subtends an angle theta at the circle’s centre.
    theta = 2.0 * asin(d / (2.0 * radius))
    # Sweep is negative if drawing clockwise.
    sweep = -theta if clockwise else theta

    # Determine the maximum segment angle so that the sagitta
    # (radius * (1 - cos(dtheta/2))) is below tolerance.
    if tolerance >= radius:
        dtheta_max = abs(sweep)
    else:
        dtheta_max = 2.0 * acos(1.0 - tolerance / radius)
        if dtheta_max > abs(sweep):
            dtheta_max = abs(sweep)

    n_segments = <int>ceil(abs(sweep) / dtheta_max)
    dtheta = sweep / n_segments

    for i in range(n_segments + 1):
        angle = start_angle + i * dtheta
        x = cx + radius * cos(angle)
        y = cy + radius * sin(angle)
        points.append(vec2(x, y))

    return points
