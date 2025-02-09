from functools import partial
import sys
import pygame
from wasabigeom import quadratic_bezier, cubic_bezier, vec2

# Initialise Pygame.
pygame.init()
width, height = 800, 1000
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Bézier Curves Demo")
clock = pygame.time.Clock()

# Define some colours.
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Set the control point radius (in pixels).
CONTROL_RADIUS = 10

# Create control points for the quadratic Bézier (top half).
quad_controls = [vec2(100.0, 150.0), vec2(400.0, 50.0), vec2(700.0, 150.0)]

# Create control points for the cubic Bézier (bottom half).
cubic_controls = [
    vec2(100.0, 350.0),
    vec2(300.0, 250.0),
    vec2(500.0, 450.0),
    vec2(700.0, 350.0),
]

shape_controls = [
    vec2(200.0, 550.0),
    vec2(400.0, 450.0),
    vec2(600.0, 550.0),
    vec2(700.0, 700.0),
    vec2(400.0, 850.0),
    vec2(100.0, 700.0)
]


def compute_closed_bezier():
    """
    Compute the full list of vec2 points for a closed quadratic Bézier curve composed
    of three segments using adaptive subdivision.
    """
    tol = 0.5  # Tolerance in metres for adaptive subdivision.
    E0, C0, E1, C1, E2, C2 = shape_controls
    pts0 = quadratic_bezier(E0, C0, E1, tol)
    pts1 = quadratic_bezier(E1, C1, E2, tol)
    pts2 = quadratic_bezier(E2, C2, E0, tol)
    # Each segment returns its starting point, so we remove duplicates.
    return pts0[:-1] + pts1[:-1] + pts2


def draw_filled_closed_bezier(surface, color):
    """
    Draw a filled closed Bézier curve on the given surface.

    The curve is computed from three quadratic Bézier segments forming a closed loop.

    Parameters:
      surface : pygame.Surface
          The surface to draw on.
      color : tuple
          The fill colour (e.g. (173, 216, 230) for light blue).
    """
    pts = compute_closed_bezier()
    # Convert each vec2 point to an (x, y) tuple.
    polygon_points = [(p.x, p.y) for p in pts]
    pygame.draw.polygon(surface, color, polygon_points)

# This variable will keep track of a control point being dragged.
# It will store a tuple: (curve, index), where curve is either "quad" or "cubic".
selected_control = None


def draw_curve(points, color):
    """Draw a curve by connecting the list of vec2 points with line segments."""
    if len(points) < 2:
        return
    pygame.draw.lines(screen, color, False, [(p.x, p.y) for p in points], 2)


def draw_control_points(controls, color):
    """Draw the control points as filled circles."""
    for p in controls:
        pygame.draw.circle(screen, color, (int(p.x), int(p.y)), CONTROL_RADIUS)


def draw_control_polygon(controls):
    """Draw the control polygon (the polyline joining the control points)."""
    line = partial(pygame.draw.aaline, screen, GRAY)
    match controls:
        case (a, b, c):
            line(a, b)
            line(b, c)
        case (a, b, c, d):
            line(a, b)
            line(c, d)
        case _:
            for i in range(len(controls) + 1)[::2]:
                a, b, c = (controls[i:] + controls[:i])[:3]
                line(a, b)
                line(b, c)


def draw():
    # Clear the screen.
    screen.fill(WHITE)

    draw_filled_closed_bezier(screen, (173, 216, 230))
    draw_control_polygon(shape_controls)
    draw_control_points(shape_controls, RED)

    # --- Draw quadratic Bézier ---
    draw_control_polygon(quad_controls)
    # Compute the quadratic Bézier curve using your adaptive subdivision function.
    quad_curve = quadratic_bezier(*quad_controls)
    draw_curve(quad_curve, BLUE)
    draw_control_points(quad_controls, RED)

    # --- Draw cubic Bézier ---
    draw_control_polygon(cubic_controls)
    # Compute the cubic Bézier curve using your adaptive subdivision function.
    cubic_curve = cubic_bezier(*cubic_controls)
    if len(cubic_curve) == 2:
        print(cubic_controls)

    draw_curve(cubic_curve, GREEN)
    draw_control_points(cubic_controls, YELLOW)
    pygame.display.flip()


def get_control_under_mouse(pos):
    """
    Return a tuple (curve, index) if the mouse is over a control point.
    Search quadratic first then cubic.
    """
    mouse = vec2(*pos)
    for controls in (quad_controls, cubic_controls, shape_controls):
        for i, p in enumerate(controls):
            if mouse.distance_to(p) < CONTROL_RADIUS:
                return (controls, i)
    return None


draw()
running = True
while running:
    dirty = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button.
                cp = get_control_under_mouse(event.pos)
                if cp is not None:
                    selected_control = cp

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                selected_control = None

        elif event.type == pygame.MOUSEMOTION:
            if selected_control is not None:
                controls, idx = selected_control
                # Update the position of the dragged control point.
                controls[idx] = vec2(event.pos)
                dirty = True

    if dirty:
        draw()
    clock.tick(60)
