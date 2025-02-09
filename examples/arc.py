import math
from wasabigeom import vec2, circular_arc, ZRect
import pygame


CONTROL_RADIUS = 8


def draw_filled_rounded_rect(surface, rect, radius):
    """
    Draw a filled rounded rectangle on surface using a pygame.Rect and a corner radius.
    The outline is computed by drawing straight lines between circular arcs computed with
    circular_arc() for each corner.
    """
    # Clamp radius to half the width/height.
    r = min(radius, rect.width/2, rect.height/2)

    rx = vec2(r, 0)
    ry = vec2(0, r)
    pts = []
    # Top edge.
    pts.append(rect.topleft + rx)
    pts.append(rect.topright - rx)

    # Top-right arc.
    arc_tr = circular_arc(pts[-1], rect.topright + ry, r, clockwise=False)
    pts.extend(arc_tr[1:])

    # Right edge.
    pts.append(rect.bottomright - ry)

    # Bottom-right arc.
    arc_br = circular_arc(pts[-1], rect.bottomright - rx, r, clockwise=False)
    pts.extend(arc_br[1:])

    # Bottom edge.
    pts.append(rect.bottomleft + rx)

    # Bottom-left arc.
    arc_bl = circular_arc(pts[-1], rect.bottomleft - ry, r, clockwise=False)
    pts.extend(arc_bl[1:])

    # Left edge.
    pts.append(rect.topleft + ry)
    # Top-left arc.
    arc_tl = circular_arc(pts[-1], pts[0], r, clockwise=False)
    pts.extend(arc_tl[1:])

    pygame.draw.polygon(surface, (200, 200, 255), pts)

def draw_controls(surface, rect, radius):
    """
    Draw small circles at the control points.
    - Red: top-left corner (controls rectangle position).
    - Green: bottom-right corner (controls rectangle size).
    - Blue: a point at (rect.left+radius, rect.top) (controls the corner radius).
    """
    pygame.draw.circle(surface, (255, 0, 0), rect.topleft, CONTROL_RADIUS)
    pygame.draw.circle(surface, (0, 255, 0), rect.bottomright, CONTROL_RADIUS)
    tr = (rect.left + int(radius), rect.top)
    pygame.draw.circle(surface, (0, 0, 255), tr, CONTROL_RADIUS)

def get_control_under_mouse(pos, rect, radius):
    """
    Return a string identifying the control point under the mouse (if any):
      "tl" for top-left, "br" for bottom-right, "tr" for top-right (radius control).
    """
    mx, my = pos
    tl = rect.topleft
    if (mx - tl[0])**2 + (my - tl[1])**2 <= CONTROL_RADIUS**2:
        return "tl"
    br = rect.bottomright
    if (mx - br[0])**2 + (my - br[1])**2 <= CONTROL_RADIUS**2:
        return "br"
    tr = (rect.left + radius, rect.top)
    if (mx - tr[0])**2 + (my - tr[1])**2 <= CONTROL_RADIUS**2:
        return "tr"
    return None

def update_rect(control, pos, rect):
    """
    Update the rectangle's position and size according to the control point being dragged.
    For "tl", the top-left is moved; for "br", the bottom-right is moved.
    """
    x, y = pos
    if control == "tl":
        new_left, new_top = x, y
        br = rect.bottomright
        rect.left = new_left
        rect.top = new_top
        rect.width = br[0] - new_left
        rect.height = br[1] - new_top
    elif control == "br":
        new_right, new_bottom = x, y
        tl = rect.topleft
        rect.width = new_right - tl[0]
        rect.height = new_bottom - tl[1]
    return rect


def update_radius(control, pos, rect):
    """
    Update the global radius based on the mouse position when dragging the top-right control.
    The new radius is the horizontal distance from rect.left to the mouse.
    """
    if control == "tr":
        new_r = pos[0] - rect.left
        new_r = max(0, new_r)
        new_r = min(new_r, rect.width/2, rect.height/2)
        return new_r
    return None


pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Rounded Rectangle Demo")
clock = pygame.time.Clock()

# Global rectangle (defined using pygame.Rect) and corner radius.
rect = ZRect(100, 100, 400, 300)
radius = 50.0

selected_control = None

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                ctrl = get_control_under_mouse(event.pos, rect, radius)
                if ctrl is not None:
                    selected_control = ctrl
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                selected_control = None
        elif event.type == pygame.MOUSEMOTION:
            if selected_control is not None:
                if selected_control in ["tl", "br"]:
                    rect = update_rect(selected_control, event.pos, rect)
                    # Ensure the radius is not larger than half the width/height.
                    radius = min(radius, rect.width/2, rect.height/2)
                elif selected_control == "tr":
                    new_r = update_radius(selected_control, event.pos, rect)
                    if new_r is not None:
                        radius = new_r

    screen.fill((255, 255, 255))
    pygame.draw.rect(screen, (0, 0, 0), rect, 1)
    draw_filled_rounded_rect(screen, rect, radius)
    draw_controls(screen, rect, radius)
    pygame.display.flip()
    clock.tick(60)
