from math import hypot
from hypothesis import given, strategies as st
from pytest import approx
from cyvec import Vector

v = Vector


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
    a = v(x, y)
    assert a.x == x
    assert a.y == y


@given(x=floats, y=floats)
def test_repr(x, y):
    """Construct a vec and display its repr."""
    a = v(x, y)
    assert eval(repr(a)) == a



@given(ax=floats, ay=floats, bx=floats, by=floats)
def test_add(ax, ay, bx, by):
    """We can add two vectors."""
    if ax + bx in (inf, -inf) or ay + by in (inf, -inf):
        return
    assert v(ax, ay) + v(bx, by) == v(ax + bx, ay + by)


def test_len():
    """The length of a vector is 2."""
    assert len(v(0, 0)) == 2


def test_iter():
    """We can unpack a vector."""
    x, y = v(1, 2)
    assert x, y == (1, 2)


@given(x=floats, y=floats)
def test_length(x, y):
    """We can get the length of a vector."""
    vec = v(x, y)
    assert vec.length() == approx(hypot(x, y))


@given(x=floats, y=floats)
def test_normalized(x, y):
    """We can normalize a vector."""
    vec = v(x, y)
    if vec.is_zero():
        return
    r, theta = vec.to_polar()
    assert vec.normalized().to_polar() == approx((1.0, theta))
