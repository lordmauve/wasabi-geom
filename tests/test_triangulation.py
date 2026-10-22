"""Coverage/indices matter; valid triangulations need not use the same diagonals."""
from array import array

import numpy as np
import pytest

from wasabigeom import triangulate


def ring_area(points):
    points = np.asarray(points, dtype=np.float64)
    # Translate before multiplying to avoid cancellation for large offsets.
    points = points - points[0]
    return abs(np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                      - points[:, 1] * np.roll(points[:, 0], -1))) / 2


def check(points, rings=None):
    indices = np.asarray(triangulate(points, ring_ends=rings))
    assert indices.dtype == np.uint32
    assert indices.ndim == 1 and len(indices) % 3 == 0
    assert not len(indices) or indices.max() < len(points)
    triangles = np.asarray(points, dtype=np.float64)[indices.reshape(-1, 3)]
    if len(triangles):
        a = triangles[:, 1] - triangles[:, 0]
        b = triangles[:, 2] - triangles[:, 0]
        areas = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
        assert np.all(areas >= 0)
        actual = areas.sum()
    else:
        actual = 0
    ends = list(rings) if rings is not None else [len(points)]
    start = 0
    expected = 0
    for i, end in enumerate(ends):
        expected += (1 if i == 0 else -1) * ring_area(points[start:end])
        start = end
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-12)
    return indices


@pytest.mark.parametrize('dtype', ['f4', 'f8'])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('vertices', [
    [(0, 0), (2, 0), (0, 2)],
    [(0, 0), (4, 0), (4, 4), (2, 1), (0, 4)],
    [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)],
    [(0, 0), (4, 0), (4, 0), (4, 4), (0, 4)],
    [(0, 0), (2, 0), (4, 0), (4, 4), (0, 4)],
])
def test_simple(vertices, dtype, reverse):
    points = np.asarray(vertices, dtype=dtype)
    if reverse:
        points = points[::-1]
    check(points)


@pytest.mark.parametrize('dtype', ['f4', 'f8'])
@pytest.mark.parametrize('outer_reverse', [False, True])
@pytest.mark.parametrize('hole_reverse', [False, True])
def test_holes(dtype, outer_reverse, hole_reverse):
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(2, 2), (4, 2), (4, 4), (2, 4)]
    if outer_reverse:
        outer.reverse()
    if hole_reverse:
        hole.reverse()
    check(np.array(outer + hole, dtype=dtype), np.array([4, 8], dtype='u4'))


def test_strides_readonly_and_precision():
    original = np.array([(0, 0), (4, 0), (4, 4), (2, 1), (0, 4)], dtype='f8')
    storage = np.ones((len(original), 4), dtype='f8')
    storage[:, ::2] = original + 1e12
    points = storage[:, ::2]
    points.flags.writeable = False
    check(points)
    check(points[::-1, ::-1])
    check(original * 1e-12)


@pytest.mark.parametrize('n', [0, 1, 2])
def test_too_few_vertices(n):
    assert len(triangulate(np.zeros((n, 2), dtype='f4'))) == 0


def test_collinear():
    assert len(triangulate(np.array([(0, 0), (1, 0), (2, 0)], dtype='f8'))) == 0


def test_buffer_without_numpy():
    points = memoryview(array('f', [0, 0, 2, 0, 0, 2])).cast('B').cast('f', (3, 2))
    result = triangulate(points)
    assert memoryview(result).format == 'I'
    assert sorted(result) == [0, 1, 2]


@pytest.mark.parametrize('rings', [[0, 4], [3], [5], [4, 4], [3, 2, 4], [-1, 4], [2.5, 4]])
def test_invalid_rings(rings):
    with pytest.raises((ValueError, TypeError, OverflowError)):
        triangulate(np.zeros((4, 2), dtype='f4'), ring_ends=rings)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite(bad):
    points = np.array([(0, 0), (1, 0), (0, bad)], dtype='f8')
    with pytest.raises(ValueError, match='finite'):
        triangulate(points)


def test_invalid_shape_dtype():
    for points in [np.zeros((4, 3), 'f4'), np.zeros(8, 'f4'), np.zeros((4, 2), 'i4')]:
        with pytest.raises((ValueError, TypeError)):
            triangulate(points)


def test_refused_polygon_does_not_return_partial_fill():
    bow_tie = np.array([(0, 0), (2, 2), (0, 2), (2, 0)], dtype='f8')
    with pytest.raises(ValueError, match='exactly'):
        triangulate(bow_tie)
    outside_hole = np.array([(0, 0), (10, 0), (10, 10), (0, 10),
                             (20, 20), (21, 20), (21, 21), (20, 21)], dtype='f8')
    with pytest.raises(ValueError, match='exactly'):
        triangulate(outside_hole, [4, 8])
