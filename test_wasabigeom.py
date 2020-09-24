from math import hypot
from hypothesis import given, strategies as st
from pytest import approx

from wasabigeom import vec2


floats = st.floats(
    allow_nan=False,
    allow_infinity=False,
    min_value=-1e100,
    max_value=1e100,
)
inf = float('inf')


@given(x=floats, y=floats)
def test_getattr(x, y):
    """We can get the x and y attributes."""
    a = vec2(x, y)
    assert a.x == x
    assert a.y == y


@given(x=floats, y=floats)
def test_repr(x, y):
    """Construct a vec and display its repr."""
    a = vec2(x, y)
    assert eval(repr(a)) == a



@given(ax=floats, ay=floats, bx=floats, by=floats)
def test_add(ax, ay, bx, by):
    """We can add two vectors."""
    if ax + bx in (inf, -inf) or ay + by in (inf, -inf):
        return
    assert vec2(ax, ay) + vec2(bx, by) == vec2(ax + bx, ay + by)


def test_len():
    """The length of a vector is 2."""
    assert len(vec2(0, 0)) == 2


def test_iter():
    """We can unpack a vector."""
    x, y = vec2(1, 2)
    assert x, y == (1, 2)


@given(x=floats, y=floats)
def test_length(x, y):
    """We can get the length of a vector."""
    vec = vec2(x, y)
    assert vec.length() == approx(hypot(x, y))


@given(x=floats, y=floats)
def test_normalized(x, y):
    """We can normalize a vector."""
    vec = vec2(x, y)
    if vec.is_zero():
        return
    r, theta = vec.to_polar()
    assert vec.normalized().to_polar() == approx((1.0, theta))


def test_normalize():
    """We can normalize a vector."""
    roothalf = 0.5 ** 0.5
    assert tuple(vec2(-1.0, 1.0).normalized()) == approx((-roothalf, roothalf))


def test_add_tuple():
    """We can add with a tuple on the right hand side."""
    assert vec2(0, 1) + (1, 0) == vec2(1, 1)


def test_radd_tuple():
    """We can add with a tuple on the left hand side."""
    assert (1, 0) + vec2(0, 1) == vec2(1, 1)


def test_sub_tuple():
    """We can subtract with a tuple on the right hand side."""
    assert vec2(1, 0) - (0, 1) == vec2(1, -1)


def test_rsub():
    """We can subtract with a tuple on the left hand side."""
    assert (1, 0) - vec2(0, 1) == vec2(1, -1)
