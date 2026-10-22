# cython: language_level=3
"""Buffer-protocol binding for the vendored Earstrip triangulator."""
from cpython.buffer cimport Py_buffer, PyObject_GetBuffer, PyBuffer_Release, PyBUF_FORMAT, PyBUF_STRIDES
from cpython cimport array
from libc.stdint cimport uint32_t, UINT32_MAX
from libc.string cimport memcpy, strcmp
from libcpp.vector cimport vector
from operator import index

cdef extern from "earstrip_bridge.hpp" namespace "wasabi":
    vector[uint32_t] triangulate32(const char*, size_t, Py_ssize_t, Py_ssize_t, const vector[uint32_t]&) except +
    vector[uint32_t] triangulate64(const char*, size_t, Py_ssize_t, Py_ssize_t, const vector[uint32_t]&) except +

cdef array.array _indices = array.array('I')
if _indices.itemsize != sizeof(uint32_t):
    raise ImportError('Earstrip requires a 32-bit unsigned int array type')


def triangulate(vertices, ring_ends=None):
    """Triangulate float32/float64 ``(N, 2)`` buffers into uint32 indices.

    ``ring_ends`` contains cumulative vertex counts (outer ring first, then
    holes); by default all vertices belong to a single outer ring. Either ring
    winding and a repeated closing vertex are supported. Strided and read-only
    buffers are accepted without copying the input. Returns an independent
    ``array.array('I')``; ``numpy.asarray(result)`` is a zero-copy uint32 view.

    Coordinates must be finite. Rings must be simple, and holes must not cross
    each other or the outer boundary. This is not a topology validator. A polygon
    Earstrip cannot finish exactly raises ValueError instead of returning a
    partial fill or introducing new vertices. Degenerate rings can return empty.
    Triangles are counter-clockwise in y-up coordinates; diagonal/order choices
    are not part of the API. No NumPy dependency is required.
    """
    cdef Py_buffer buf
    cdef vector[uint32_t] ends
    cdef vector[uint32_t] result
    cdef Py_ssize_t n, previous = 0, end, i
    cdef bint single_precision
    cdef array.array output
    PyObject_GetBuffer(vertices, &buf, PyBUF_FORMAT | PyBUF_STRIDES)
    try:
        if buf.ndim != 2 or buf.shape[1] != 2:
            raise ValueError('vertices must have shape (N, 2)')
        if buf.suboffsets != NULL:
            for i in range(buf.ndim):
                if buf.suboffsets[i] >= 0:
                    raise ValueError('indirect buffers are not supported')
        if strcmp(buf.format, 'f') == 0 and buf.itemsize == sizeof(float):
            single_precision = True
        elif strcmp(buf.format, 'd') == 0 and buf.itemsize == sizeof(double):
            single_precision = False
        else:
            raise TypeError('vertices must be a native-endian float32 or float64 buffer')
        n = buf.shape[0]
        if <size_t>n > UINT32_MAX:
            raise OverflowError('too many vertices for uint32 indices')
        if ring_ends is not None:
            for value in ring_ends:
                end = index(value)
                if end <= previous or end > n:
                    raise ValueError('ring_ends must increase strictly and end at N')
                ends.push_back(<uint32_t>end)
                previous = end
            if previous != n:
                raise ValueError('ring_ends must end at N')
        if single_precision:
            result = triangulate32(<const char*>buf.buf, n, buf.strides[0], buf.strides[1], ends)
        else:
            result = triangulate64(<const char*>buf.buf, n, buf.strides[0], buf.strides[1], ends)
        output = array.clone(_indices, result.size(), False)
        if result.size():
            memcpy(output.data.as_voidptr, result.data(), result.size() * sizeof(uint32_t))
        return output
    finally:
        PyBuffer_Release(&buf)
