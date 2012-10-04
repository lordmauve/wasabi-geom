from .vector import Vector, cached, unit_x, unit_y


__all__ = [
    'Line',
    'LineSegment',
    'Projection',
    'Segment',
    'PolyLine',
    'x_axis',
    'y_axis'
]


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
            `direction` : Vector
                A (non-zero) vector perpendicular to the line.
            `distance` : float
                The distance from the origin to the line.

        """
        if not isinstance(direction, Vector):
            direction = Vector(direction)
        self.direction = direction.normalised()
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
            `first`, `second` : Vector
                The vectors used to construct the line.

        """
        if not isinstance(first, Vector):
            first = Vector(first)
        along = (second - first).normalised()
        direction = -along.perpendicular()
        distance = first.dot(direction)
        return cls(direction, distance)

    @cached
    def offset(self):
        """The projection of the origin onto the line.

        """
        return self.direction * self.distance

    def project(self, point):
        """Compute the projection of a point onto the line.

        :Parameters:
            `point` : Vector
                The point to project onto the line.

        """
        parallel = self.along.project(point)
        return parallel + self.offset

    def reflect(self, point):
        """Reflect a point in the line.

        :Parameters:
            `point` : Vector
                The point to reflect in the line.

        """
        if not isinstance(point, Vector):
            point = Vector(point)
        offset_distance = point.dot(self.direction) - self.distance
        return point - 2 * self.direction * offset_distance

    mirror = reflect

    def distance_to(self, point):
        """Return the (signed) distance to a point.

        :Parameters:
            `point` : Vector
                The point to measure the distance to.

        """
        if not isinstance(point, Vector):
            point = Vector(point)
        return point.dot(self.direction) - self.distance

    altitude = distance_to

    def is_on_left(self, point):
        """Determine if the given point is left of the line.

        :Parameters:
            `point` : Vector
                The point to locate.

        """
        return self.distance_to(point) < 0

    is_inside = is_on_left

    def is_on_right(self, point):
        """Determine if the given point is right of the line.

        :Parameters:
            `point` : Vector
                The point to locate.

        """
        return self.distance_to(point) > 0

    def parallel(self, point):
        """Return a line parallel to this one through the given point.

        :Parameters:
            `point` : Vector
                The point through which to trace a line.

        """
        if not isinstance(point, Vector):
            point = Vector(point)
        distance = point.dot(self.direction)
        return Line(self.direction, distance)

    def perpendicular(self, point):
        """Return a line perpendicular to this one through the given point. The
        orientation of the line is consistent with ``Vector.perpendicular``.

        :Parameters:
            `point` : Vector
                The point through which to trace a line.

        """
        if not isinstance(point, Vector):
            point = Vector(point)
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
            `first`, `second` : Vector
                The vectors used to construct the line.

        """
        if not isinstance(first, Vector):
            first = Vector(first)
        if not isinstance(second, Vector):
            second = Vector(second)
        line = Line.from_points(first, second)
        d1, d2 = first.dot(line.along), second.dot(line.along)
        return cls(line, min(d1, d2), max(d1, d2))

    @cached
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

    @cached
    def start(self):
        """One endpoint of the line segment (corresponding to 'min_dist').

        """
        start, end = self._endpoints()
        self.mid = (start + end) / 2
        self.end = end
        return start

    @cached
    def mid(self):
        """The midpoint of the line segment.

        """
        start, end = self._endpoints()
        self.start = start
        self.end = end
        return (start + end) / 2

    @cached
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
            `point` : Vector
                The point to minimise the distance to.

        """
        if not isinstance(point, Vector):
            point = Vector(point)
        distance = point.dot(self.line.along)
        if distance >= self.max_dist:
            return self.end
        elif distance <= self.min_dist:
            return self.start
        return self.line.along * distance + self.line.offset

    def distance_to(self, point):
        """Return the shortest distance to a given point.

        :Parameters:
            `point` : Vector
                The point to measure the distance to.

        """
        if not isinstance(point, Vector):
            point = Vector(point)
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
        self.edge = (p2 - p1).normalised()
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
