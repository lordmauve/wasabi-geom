from math import hypot, pi

from hypothesis import given, strategies as st
from pytest import approx, raises
import pytest
import numpy as np

from wasabigeom import vec2, Transform


tau = 2 * pi


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


def test_mul_float():
    """We can multiply a vector by a float."""
    assert vec2(10.0, 5.0) * 2.0 == vec2(20, 10)


def test_mul_int():
    """We can multiply a vector by an integer."""
    assert vec2(10.0, 5.0) * 2 == vec2(20, 10)


def test_rmul_float():
    """We can multiply a vector by a float."""
    assert 2.0 * vec2(10.0, 5.0) == vec2(20, 10)


def test_rmul_int():
    """We can multiply a vector by an integer."""
    assert 2 * vec2(10.0, 5.0) == vec2(20, 10)


def test_add_notimplemented():
    """Adding to an unsupported type returns NotImplemented."""
    assert vec2(0, 0).__add__(None) == NotImplemented


def test_sub_int():
    """It is a TypeError to subtract a vector and an int."""
    with raises(TypeError):
        vec2(0, 0) - 1


def test_rsub_int():
    """It is a TypeError to subtract a vector and an int."""
    with raises(TypeError):
        1 - vec2(0, 0)


def test_add_invalid_tuple_length():
    """It is an error to add a vector and a tuple of length != 2."""
    with raises(TypeError):
        vec2(0, 0) + ()


def test_add_none_tuple():
    """It is an error to vectors that don't contain numbers."""
    with raises(TypeError):
        vec2(0, 0) + (None, None)


def test_div():
    """We can divide a vec2."""
    assert vec2(10, 20) / 2 == vec2(5, 10)


def test_rdiv():
    """We can divide by a vec2."""
    assert 8 / vec2(2, 4) == vec2(4, 2)


def test_negate():
    """We can negate a vector."""
    assert -vec2(1, -2) == vec2(-1, 2)


@given(angle=st.floats(min_value=-10 * tau, max_value=10 * tau))
def test_rotate(angle):
    """We can rotate a vector."""
    r, theta = vec2(1, 1).rotated(angle).to_polar()
    assert theta % tau == approx((angle + tau / 8) % tau)
    assert r == approx(2 ** 0.5)


def test_from_polar():
    """We can construct a vector from polar coordinates."""
    assert vec2.from_polar(2, pi) == approx((-2, 0))


def test_construct_from_array():
    """We can construct a vector by passing any sequence."""
    from array import array
    a = array('d', [6.0, 5.0])
    assert vec2(a) == vec2(6, 5)


def test_build_transform():
    """We can build a transformation matrix from translate/rotate/scale."""
    t = Transform.build(xlate=(3, 4), rot=pi / 4, scale=(2, 1))

    root2 = 2 ** 0.5
    np.testing.assert_array_almost_equal(
        np.asarray(t),
        np.array([
            [root2 , root2 / -2, 3],
            [root2, root2 / 2, 4],
        ])
    )


def test_chain_transforms():
    """We can chain together transformations by multiplying them."""
    r = Transform.build(xlate=(1, 2), rot=0.5) * Transform.build(rot=-0.5)
    np.testing.assert_array_almost_equal(
        np.asarray(r),
        np.asarray(Transform(1., 0., 1.,
                             0., 1., 2.))
    )


def test_inverse():
    """We can find the inverse transform."""
    r = Transform.build(rot=0.5, scale=(2, 2))
    invr = r.inverse()
    np.testing.assert_array_almost_equal(
        np.asarray(r * invr),
        np.asarray(Transform(1., 0., 0.,
                             0., 1., 0.))
    )


for_np_float_types = pytest.mark.parametrize('float_type', [np.float32, np.float64])


@for_np_float_types
def test_xlate(float_type):
    """We can translate coordinates using Transform."""
    t = Transform.build(xlate=(1, -1))
    coords = np.array([
        (0, 0),
        (1, 1),
        (-1, -2),
    ], dtype=float_type)
    np.testing.assert_array_equal(
        t.transform(coords),
        np.array([
            (1, -1),
            (2, 0),
            (0, -3),
        ])
    )


@for_np_float_types
def test_xform_inplace(float_type):
    """We can transform coordinates in place."""
    t = Transform.build(xlate=(1, -1), rot=pi / 2, scale=(2, 2))
    coords = np.array([
        (0, 0),
        (1, 1),
        (-1, -2),
    ], dtype=float_type)

    t.transform(coords, coords)
    np.testing.assert_array_almost_equal(
        coords,
        np.array([
            (1, -1),
            (-1, 1),
            (5, -3),
        ])
    )
