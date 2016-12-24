#pragma once

#include "libcgt/core/vecmath/Matrix3f.h"
#include "libcgt/core/vecmath/Vector3f.h"
#include "libcgt/core/vecmath/Vector4f.h"

// A plane in 3D, defined by Ax + By + Cz + D = 0
class Plane3f
{
public:

    float a = 0;
    float b = 0;
    float c = 0;
    float d = 0;

    // The XY plane, with the normal pointing up (0,0,1)
    static Plane3f XY();

    // The YZ plane, with the normal pointing out of the page (1,0,0)
    static Plane3f YZ();

    // The ZX plane, with the normal pointing right (0,1,0)
    static Plane3f ZX();

    Plane3f() = default;

    // Construct a plane directly from its coefficients
    Plane3f( float a, float b, float c, float d );
    Plane3f( const Plane3f& p ) = default;
    Plane3f& operator = ( const Plane3f& p ) = default;

    // Construct the plane defined by 3 (distinct) points.
    Plane3f( const Vector3f& p0, const Vector3f& p1, const Vector3f& p2 );

    // Construct a plane given a point on the plane and its (not necessarily
    // unit length) normal. It will be normalized.
    Plane3f( const Vector3f& p, const Vector3f& normal );

    // Return (a,b,c).
    Vector3f normal() const;

    // Returns normal().normalized()
    Vector3f unitNormal() const;

    // Projects the point p to the point q closest to p on the plane.
    Vector3f closestPointOnPlane( const Vector3f& p ) const;

    // Returns the *signed* shortest distance between p and the plane.
    // distance is positive on the same side as the normal
    float distance( const Vector3f& p ) const;

    // Returns the point on the plane closest to the origin
    // http://en.wikipedia.org/wiki/Point_on_plane_closest_to_origin
    Vector3f pointOnPlane() const;

    // Returns an orthonormal basis u, v, and n on the plane
    // given a *preferred* u vector. If u is not on the plane,
    // then u is projected onto the plane first.
    // Returns the matrix [ u v n ]
    Matrix3f basis( const Vector3f& u ) const;

    // Returns an orthonormal basis u, v, n on the plane as the matrix
    // [ u v n ].
    Matrix3f basis() const;

    // Returns the same plane, but with its normal flipped
    Plane3f flipped() const;

    // Returns a plane parallel this this at distance z in the direction of the
    // normal.
    Plane3f offset( float z ) const;

    static float cosineDihedralAngle( const Plane3f& p0, const Plane3f& p1 );
};
