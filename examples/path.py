import math
import sys
import pygame
from wasabigeom import vec2, quadratic_bezier, cubic_bezier, arc_to_cubic_beziers, Transform


class Path:
    def __init__(self):
        # Each segment is stored as a tuple. The first element is a command code:
        # 'M': move‐to (start a new subpath), followed by one vec2.
        # 'L': line‐to, followed by the destination vec2.
        # 'Q': quadratic Bézier, followed by (ctrl, dest).
        # 'C': cubic Bézier, followed by (ctrl1, ctrl2, dest).
        self.segments = []
        self._start = None   # starting point of the current subpath (vec2)
        self.current = None  # current point (vec2)
        self.color = (128, 128, 128)
        self.position = vec2(0, 0)
        self.angle = 0
        self.scale = (1, 1)

    def start(self, pos: vec2) -> None:
        """Begin a new subpath at pos."""
        self._start = pos
        self.current = pos
        self.segments.append(('M', pos))

    def line_to(self, pos: vec2) -> None:
        """Add a straight line from the current point to pos."""
        self.segments.append(('L', pos))
        self.current = pos

    def quad_to(self, ctrl: vec2, pos: vec2) -> None:
        """Add a quadratic Bézier curve from the current point to pos, using ctrl as the control point."""
        self.segments.append(('Q', ctrl, pos))
        self.current = pos

    def cubic_to(self, ctrl1: vec2, ctrl2: vec2, pos: vec2) -> None:
        """Add a cubic Bézier curve from the current point to pos, using ctrl1 and ctrl2 as control points."""
        self.segments.append(('C', ctrl1, ctrl2, pos))
        self.current = pos

    def arc_to(self, pos: vec2, radius: float, clockwise: bool, tolerance: float = 0.5) -> None:
        """
        Add an arc from the current point to pos, with the given radius and direction.
        The arc is converted to one or more cubic Bézier segments (using arc_to_cubic_beziers)
        and appended to the segment list.
        """
        # Our arc is defined by the current point (p0) and pos (p1).
        p0 = self.current
        p1 = pos
        sep = p1 - p0
        d2 = sep.length_squared()
        if d2 > 4 * radius * radius:
            raise ValueError("Chord length exceeds diameter for the given radius")
        if d2 == 0.0:
            return  # nothing to do
        d = math.sqrt(d2)
        # Compute chord midpoint
        m = p0 + sep * 0.5
        # h = √(radius² - (d/2)²)
        h = math.sqrt(radius * radius - (d / 2.0) ** 2)
        # Determine the unit perpendicular vector. (For a clockwise arc, use the right‐hand perpendicular.)
        if clockwise:
            perp = vec2(sep.y / d, -sep.x / d)
        else:
            perp = vec2(-sep.y / d, sep.x / d)
        # Compute circle center.
        center = m + perp * h
        # Starting angle (angle from center to p0)
        start_angle = math.atan2(p0.y - center.y, p0.x - center.x)
        # The chord subtends an angle theta.
        theta = 2.0 * math.asin(d / (2.0 * radius))
        sweep = -theta if clockwise else theta
        end_angle = start_angle + sweep
        # Convert the arc to cubic Bézier segments.
        # arc_to_cubic_beziers returns a list of segments, each a tuple (P0, P1, P2, P3).
        segments = arc_to_cubic_beziers((center.x, center.y), radius, start_angle, end_angle)
        for seg in segments:
            # We ignore the first point of the first segment (which should equal self.current)
            # and store each cubic segment.
            self.segments.append(('C', seg[1], seg[2], seg[3]))
            self.current = seg[3]

    def close(self) -> None:
        """Close the current subpath by adding a line to the starting point, if needed."""
        if self._start is not None and self.current != self._start:
            self.line_to(self._start)

    def _to_poly(self, tolerance: float) -> list:
        """
        Convert the path into a polygon (list of vec2 points) by subdividing all curves
        using the given tolerance. Lines are added directly.
        For quadratic segments, quadratic_bezier() is called;
        for cubic segments, cubic_bezier() is called.
        """
        poly = []
        current = None
        t = Transform.build(self.position, self.angle, self.scale)
        for seg in self.segments:
            cmd = seg[0]
            if cmd == 'M':
                current = t * seg[1]
                poly.append(current)
            elif cmd == 'L':
                current = t * seg[1]
                poly.append(current)
            elif cmd == 'Q':
                # seg = ('Q', ctrl, end)
                pts = quadratic_bezier(current, t * seg[1], t * seg[2], tolerance)
                # Avoid duplicating the start point.
                poly.extend(pts[1:] if poly else pts)
                current = t * seg[2]
            elif cmd == 'C':
                # seg = ('C', ctrl1, ctrl2, end)
                pts = cubic_bezier(current, t * seg[1], t * seg[2], t * seg[3], tolerance)
                poly.extend(pts[1:] if poly else pts)
                current = t * seg[3]
        return poly

    def draw(self):
        """Draw the polygon on screen as a filled shape."""
        poly = self._to_poly(tolerance=0.5)
        if len(poly) < 2:
            return
        pygame.draw.polygon(screen, self.color, poly)

    @classmethod
    def rect(cls, rect: pygame.Rect) -> "Path":
        """Return a new Path representing the outline of the given rectangle."""
        w = rect.width
        h = rect.height
        x = vec2(w / 2, 0)
        y = vec2(0, h / 2)
        p = cls()
        p.start(-x - y)
        p.line_to(x - y)
        p.line_to(x + y)
        p.line_to(-x + y)
        p.close()
        p.position = vec2(*rect.center)
        return p

    @classmethod
    def ngon(cls, center: vec2, radius: float, n: int, start_angle: float = 0.0) -> "Path":
        """
        Return a new Path representing a regular n-gon.
        The vertices are computed using the given center, radius, and start_angle.
        """
        p = cls()
        angle_step = 2 * math.pi / n
        for i in range(n):
            angle = start_angle + i * angle_step
            pt = vec2(
                radius * math.cos(angle),
                radius * math.sin(angle)
            )
            if i == 0:
                p.start(pt)
            else:
                p.line_to(pt)
        p.close()
        p.position = center
        return p

    @classmethod
    def rounded_rect(cls, rect: pygame.Rect, radius: float, tolerance: float = 0.5) -> "Path":
        """
        Return a new Path representing a rounded rectangle.
        The rectangle is defined by the given pygame.Rect, and each corner is rounded
        with the specified radius.
        """
        w = rect.width
        h = rect.height
        x = vec2(w / 2, 0)
        y = vec2(0, h / 2)
        topleft = -x - y
        topright = x - y
        bottomright = x + y
        bottomleft = -x + y
        # Clamp radius so it does not exceed half the rect's dimensions.
        r = min(radius, rect.width / 2, rect.height / 2)
        rx = vec2(r, 0)
        ry = vec2(0, r)
        p = cls()
        p.start(topleft + rx)
        p.line_to(topright - rx)
        p.arc_to(topright + ry, r, clockwise=False, tolerance=tolerance)
        p.line_to(bottomright - ry)
        p.arc_to(bottomright - rx, r, clockwise=False, tolerance=tolerance)
        p.line_to(bottomleft + rx)
        p.arc_to(bottomleft - ry, r, clockwise=False, tolerance=tolerance)
        p.line_to(topleft + ry)
        p.arc_to(topleft + rx, r, clockwise=False, tolerance=tolerance)
        p.close()
        p.position = vec2(*rect.center)
        return p

    @classmethod
    def circle(cls, center: vec2, radius: float, tolerance: float = 0.5) -> "Path":
        """
        Return a new Path representing a circle.
        The circle is approximated with 4 cubic Bézier segments (using arc_to).
        """
        p = cls()
        rx = vec2(radius, 0)
        ry = vec2(0, radius)
        p.start(-rx)
        p.arc_to(ry, radius, clockwise=True, tolerance=tolerance)
        p.arc_to(rx, radius, clockwise=True, tolerance=tolerance)
        p.arc_to(-ry, radius, clockwise=True, tolerance=tolerance)
        p.arc_to(-rx, radius, clockwise=True, tolerance=tolerance)
        p.position = vec2(*center)
        return p

    @classmethod
    def star(cls, center: vec2, outer_radius: float, inner_radius: float,
             num_points: int, roundness: float = 0.0, tolerance: float = 0.5) -> "Path":
        """
        Return a new Path representing a star.
        The star is defined by the center, outer and inner radii, and the number of points.
        If roundness > 0 the sharp corners will be replaced by arcs of the given roundness.
        """
        # Compute 2*num_points vertices (alternating outer and inner).
        vertices = []
        angle_step = math.pi / num_points  # half the central angle per point
        for i in range(2 * num_points):
            angle = i * angle_step
            r = outer_radius if i % 2 == 0 else inner_radius
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            vertices.append(vec2(x, y))
        p = cls()
        n = len(vertices)
        if roundness <= 0:
            p.start(vertices[0])
            for pt in vertices[1:]:
                p.line_to(pt)
            p.close()
        else:
            # For each vertex, compute offset points for rounding.
            # Each vertex becomes (p0, current, p1) where p0 and p1 lie along the incoming
            # and outgoing edges at a distance equal to roundness.
            rounded = []
            for i in range(n):
                prev = vertices[i - 1]
                curr = vertices[i]
                next = vertices[(i + 1) % n]
                # Compute unit vectors from curr to prev and curr to next.
                vprev = prev - curr
                vnext = next - curr
                uprev = vprev.normalized()
                unext = vnext.normalized()
                if i % 2 == 0:
                    offset = roundness * (outer_radius / inner_radius)
                else:
                    offset = roundness / (outer_radius / inner_radius)
                p0 = curr + uprev * offset
                p1 = curr + unext * offset
                rounded.append((p0, curr, p1))
            # Build the path using the rounded corners.
            # Start at the outgoing offset of the last vertex.
            p.start(rounded[-1][2])
            for i, (p0, curr, p1) in enumerate(rounded):
                p.line_to(p0)
                p.arc_to(p1, roundness, clockwise=i % 2, tolerance=tolerance)
            p.close()
        p.position = vec2(*center)
        return p


# Create a plain rectangle from a pygame.Rect:
rect_path = Path.rect(pygame.Rect(50, 50, 200, 150))
rect_path.color = (255, 200, 128)

# Create a regular pentagon:
pentagon = Path.ngon(vec2(300, 300), 100, 5)
pentagon.color = (128, 200, 255)

# Create a rounded rectangle:
rrect = Path.rounded_rect(pygame.Rect(400, 100, 250, 200), radius=30)
rrect.color = (164, 225, 128)

# Create a circle:
circle = Path.circle(vec2(150, 450), 75)
circle.color = (255, 128, 128)
circle.scale = (1.5, 1)

# Create a star with rounded points:
star = Path.star(vec2(500, 450), outer_radius=100, inner_radius=50,
                 num_points=5, roundness=10)
star.color = (255, 128, 255)


SHAPES = (rect_path, pentagon, rrect, circle, star)

# Draw the paths on the screen:
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()
t = 0
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    screen.fill((255, 255, 255))
    # Draw the paths
    for i, poly in enumerate(SHAPES):
        poly.angle = 0.1 * math.sin(t + i)
    for poly in SHAPES:
        poly.draw()
    pygame.display.flip()
    dt = clock.tick(60)
    t += dt / 1000
