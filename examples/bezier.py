from functools import partial
import sys
import pygame
from wasabigeom import quadratic_bezier, cubic_bezier, vec2

# Initialise Pygame.
pygame.init()
width, height = 800, 600
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
    vec2(100.0, 450.0),
    vec2(300.0, 350.0),
    vec2(500.0, 550.0),
    vec2(700.0, 450.0),
]

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


def draw():
    # Clear the screen.
    screen.fill(WHITE)

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
    mx, my = pos
    for i, p in enumerate(quad_controls):
        if (p.x - mx) ** 2 + (p.y - my) ** 2 <= CONTROL_RADIUS**2:
            return (quad_controls, i)
    for i, p in enumerate(cubic_controls):
        if (p.x - mx) ** 2 + (p.y - my) ** 2 <= CONTROL_RADIUS**2:
            return (cubic_controls, i)
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
