# Copyright (c) 2009 The Super Effective Team (www.supereffective.org).
# Copyright (c) Daniel Pope
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name(s) of the copyright holders nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AS IS AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from __future__ import division

import math


def cached(func):
    """Decorate a function as a caching property.

    :Parameters:
        `func` : function
            The getter function to decorate.

    """
    cached_name = "_cached_%s" % func.func_name

    # The keywords 'getattr' and 'cached_name' are used to optimise the common
    # case (return cached value) by bringing the used names to local scope.
    def fget(self, getattr=getattr, cached_name=cached_name):
        try:
            return getattr(self, cached_name)
        except AttributeError:
            value = func(self)
            setattr(self, cached_name, value)
            return value

    def fset(self, value):
        assert not hasattr(self, cached_name)
        setattr(self, cached_name, value)

    fget.func_name = "get_" + func.func_name
    fset.func_name = "set_" + func.func_name

    return property(fget, fset, doc=func.func_doc)


class Vector(tuple):
    """Two-dimensional float vector implementation.

    """

    def __str__(self):
        """Construct a concise string representation.

        """
        return "Vector((%.2f, %.2f))" % self

    def __repr__(self):
        """Construct a precise string representation.

        """
        return "Vector((%r, %r))" % self

    @property
    def x(self):
        """The horizontal coordinate.

        """
        return self[0]

    @property
    def y(self):
        """The vertical coordinate.

        """
        return self[1]

    @cached
    def length(self):
        """The length of the vector.

        """
        return math.sqrt(self.length2)

    @cached
    def length2(self):
        """The square of the length of the vector.

        """
        vx, vy = self
        return vx ** 2 + vy ** 2

    @cached
    def angle(self):
        """The angle the vector makes to the positive x axis in the range
        (-180, 180].

        """
        vx, vy = self
        return math.degrees(math.atan2(vy, vx))

    @property
    def is_zero(self):
        """Flag indicating whether this is the zero vector.

        """
        return self.length2 < 1e-9

    def __add__(self, other):
        """Add the vectors componentwise.

        :Parameters:
            `other` : Vector
                The object to add.

        """
        return Vector((self[0] + other[0], self[1] + other[1]))

    def __radd__(self, other):
        """Add the vectors componentwise.

        :Parameters:
            `other` : Vector
                The object to add.

        """
        return Vector((other[0] + self[0], other[1] + self[1]))

    def __sub__(self, other):
        """Subtract the vectors componentwise.

        :Parameters:
            `other` : Vector
                The object to subtract.

        """
        return Vector((self[0] - other[0], self[1] - other[1]))

    def __rsub__(self, other):
        """Subtract the vectors componentwise.

        :Parameters:
            `other` : Vector
                The object to subtract.

        """
        return Vector((other[0] - self[0], other[1] - self[1]))

    def __mul__(self, other):
        """Either multiply the vector by a scalar or compute the dot product
        with another vector.

        :Parameters:
            `other` : Vector or float
                The object by which to multiply.

        """
        try:
            other = float(other)
            return Vector((self[0] * other, self[1] * other))
        except TypeError:
            return self[0] * other[0] + self[1] * other[1]

    def __rmul__(self, other):
        """Either multiply the vector by a scalar or compute the dot product
        with another vector.

        :Parameters:
            `other` : Vector or float
                The object by which to multiply.

        """
        try:
            other = float(other)
            return Vector((other * self[0], other * self[1]))
        except TypeError:
            return other[0] * self[0] + other[1] * self[1]

    def __div__(self, other):
        """Divide the vector by a scalar.

        :Parameters:
            `other` : float
                The object by which to divide.

        """
        return Vector((self[0] / other, self[1] / other))

    def __truediv__(self, other):
        """Divide the vector by a scalar.

        :Parameters:
            `other` : float
                The object by which to divide.

        """
        return Vector((self[0] / other, self[1] / other))

    def __floordiv__(self, other):
        """Divide the vector by a scalar, rounding down.

        :Parameters:
            `other` : float
                The object by which to divide.

        """
        return Vector((self[0] // other, self[1] // other))

    def __neg__(self):
        """Compute the unary negation of the vector.

        """
        return Vector((-self[0], -self[1]))

    def rotated(self, angle):
        """Compute the vector rotated by an angle.

        :Parameters:
            `angle` : float
                The angle (in degrees) by which to rotate.

        """
        vx, vy = self
        angle = math.radians(angle)
        ca, sa = math.cos(angle), math.sin(angle)
        return Vector((vx * ca - vy * sa, vx * sa + vy * ca))

    def scaled_to(self, length):
        """Compute the vector scaled to a given length.

        :Parameters:
            `length` : float
                The length to which to scale.

        """
        vx, vy = self
        s = length / self.length
        v = Vector((vx * s, vy * s))
        v.length = length
        return v

    def safe_scaled_to(self, length):
        """Compute the vector scaled to a given length, or just return the
        vector if it was the zero vector.

        :Parameters:
            `length` : float
                The length to which to scale.

        """
        if self.is_zero:
            return self
        return self.scaled_to(length)

    def normalised(self):
        """Compute the vector scaled to unit length.

        """
        vx, vy = self
        l = self.length
        v = Vector((vx / l, vy / l))
        v.length = 1.0
        return v

    normalized = normalised

    def safe_normalised(self):
        """Compute the vector scaled to unit length, or some unit vector
        if it was the zero vector.

        """
        if self.is_zero:
            return Vector((0, 1))
        return self.normalised() 

    safe_normalized = safe_normalised

    def perpendicular(self):
        """Compute the perpendicular.

        """
        vx, vy = self
        return Vector((-vy, vx))

    def dot(self, other):
        """Compute the dot product with another vector.

        :Parameters:
            `other` : Vector
                The vector with which to compute the dot product.

        """
        return self[0] * other[0] + self[1] * other[1]

    def cross(self, other):
        """Compute the cross product with another vector.

        :Parameters:
            `other` : Vector
                The vector with which to compute the cross product.

        """
        return self[0] * other[1] - self[1] * other[0]

    def project(self, other):
        """Compute the projection of another vector onto this one.

        :Parameters:
            `other` : Vector
                The vector of which to compute the projection.

        """
        return self * self.dot(other) / self.dot(self)

    def angle_to(self, other):
        """Compute the angle made to another vector in the range [0, 180].

        :Parameters:
            `other` : Vector
                The vector with which to compute the angle.

        """
        if not isinstance(other, Vector):
            other = Vector(other)
        a = abs(other.angle - self.angle)
        return min(a, 360 - a)

    def signed_angle_to(self, other):
        """Compute the signed angle made to another vector in the range
        (-180, 180].

        :Parameters:
            `other` : Vector
                The vector with which to compute the angle.

        """
        if not isinstance(other, Vector):
            other = Vector(other)
        a = other.angle - self.angle
        return min(a + 360, a, a - 360, key=abs)

    def distance_to(self, other):
        """Compute the distance to another point vector.

        :Parameters:
            `other` : Vector
                The point vector to which to compute the distance.

        """
        return (other - self).length



class Matrix(object):
    """A 2x2 matrix.

    This can be used to optimise a transform (such as a rotation) on multiple
    vectors without recomputing terms.

    To transform a vector with this matrix, use premultiplication, ie. for
    Matrix M and Vector v, ::

        t = M * v

    

    """
    __slots__ = ('x11', 'x12', 'x21', 'x22')

    def __init__(self, x11, x12, x21, x22):
        self.x11 = x11
        self.x12 = x12
        self.x21 = x21
        self.x22 = x22
    
    def __mul__(self, vec):
        """Multiple a vector by this matrix."""
        return Vector((
            self.x11 * vec.x + self.x12 * vec.y,
            self.x21 * vec.x + self.x22 * vec.y
        ))

    @staticmethod
    def rotation(angle):
        """A rotation matrix for angle a."""
        sin = math.sin(angle)
        cos = math.cos(angle)
        return Matrix(cos, -sin, sin, cos)



def v(*args):
    """Construct a vector from an iterable or from multiple arguments. Valid
    forms are therefore: ``v((x, y))`` and ``v(x, y)``.

    """
    if len(args) == 1:
        return Vector(args[0])
    return Vector(args)


#: The zero vector.
zero = Vector((0, 0))

#: The unit vector on the x-axis.
unit_x = Vector((1, 0))

#: The unit vector on the y-axis.
unit_y = Vector((0, 1))
