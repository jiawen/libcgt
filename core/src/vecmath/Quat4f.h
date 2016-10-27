#pragma once

class Vector3f;
class Vector4f;

#include "Matrix3f.h"

// q = w + x * i + y * j + z * k
// with i^2 = j^2 = k^2 = -1
class Quat4f
{
public:

    static const Quat4f ZERO;
    static const Quat4f IDENTITY;

    // Construct a quaternion from an axis-angle representation, where
    // axisAngle.norm() is the rotation angle in radians and
    // axisAngle.normalized() is the direction.
    static Quat4f fromAxisAngle( const Vector3f& axisAngle );

    // Given a rotation matrix m, returns its unit quaternion representation
    static Quat4f fromRotationMatrix( const Matrix3f& m );

    // Convert an orthonormal basis of R^3: (x, y, z = x cross y) into a
    // rotation matrix m = [x y z] and then into a unit quaternion.
    //
    // If x, y, z are the world-space vectors denoting the right, up, and back
    // vectors of a camera, then q (and m) is the mapping worldFromCamera.
    static Quat4f fromRotatedBasis(
        const Vector3f& x, const Vector3f& y, const Vector3f& z );

    // returns a unit quaternion that's a uniformly distributed rotation
    // given u[i] is a uniformly distributed random number in [0,1]
    // taken from Graphics Gems II
    static Quat4f randomRotation( float u0, float u1, float u2 );

    Quat4f();
    Quat4f( float _w, float _x, float _y, float _z );
    // returns a quaternion with 0 real part
    Quat4f( const Vector3f& v );

    // Copies the components of a Vector4f directly into this quaternion.
    Quat4f( const Vector4f& v );

    // [_w, _v.x, _v.y, _v.z].
    Quat4f( float _w, const Vector3f& _v );

    Quat4f( const Quat4f& q );
    Quat4f& operator = ( const Quat4f& q );

    // Returns the ith element.
    const float& operator [] ( int i ) const;
    float& operator [] ( int i );

    Vector3f xyz() const;
    Vector4f wxyz() const;

    float norm() const;
    float normSquared() const;
    void normalize();
    Quat4f normalized() const;

    Quat4f conjugated() const;

    // Recall that the inverse of a quaternion q is conj(q) / norm(q).
    // If q is a unit quaternion, its inverse is its conjugate.
    Quat4f inverse() const;

    // log and exponential maps
    Quat4f log() const;
    Quat4f exp() const;

    // returns unit vector for rotation and radians about the unit vector
    // The angle is guaranteed to be in [0,pi]
    // Returns axis = (0,0,0) and angle = 0 for the identity quaternion (1,0,0,0)
    Vector3f getAxisAngle( float* radiansOut ) const;

    // returns a Vector4f, with xyz = axis, and w = angle
    // axis is a unit vector, angle is in radians
    // The angle is guaranteed to be in [0,pi]
    // Returns axis = (0,0,0) and angle = 0 for the identity quaternion (1,0,0,0)
    Vector4f getAxisAngle() const;

    // If this is a unit quaternion, rotates a vector v as:
    // v' = q * quat(v) * conj(q), where quat(v) is the quaternion [0 v.xyz].
    Vector3f rotateVector( const Vector3f& v );

    // ---- Utility ----
    void print();

    // Quaternion dot product, computed just like for vectors.
    static float dot( const Quat4f& q0, const Quat4f& q1 );

    // Linear (stupid) interpolation.
    static Quat4f lerp( const Quat4f& q0, const Quat4f& q1, float alpha );

    // spherical linear interpolation
    static Quat4f slerp( const Quat4f& a, const Quat4f& b, float t,
        bool allowFlip = true );

    // Spherical quadratic interpolation between a and b at point t.
    // The quaternion tangents tanA and tanB can be computed using
    // squadTangent().
    static Quat4f squad( const Quat4f& a, const Quat4f& tanA,
        const Quat4f& tanB, const Quat4f& b,
        float t );

    // Computes a tangent at center, defined by the before and after quaternions
    // Useful for squad()
    static Quat4f squadTangent( const Quat4f& before, const Quat4f& center,
        const Quat4f& after );

    // spherical cubic interpolation: given control points q[i] and parameter t
    // computes the interpolated quaternion
    static Quat4f cubicInterpolate(
        const Quat4f& q0, const Quat4f& q1, const Quat4f& q2, const Quat4f& q3,
        float t );

    // Log-difference between a and b, used for squadTangent
    // returns log( a^-1 b )
    static Quat4f logDifference( const Quat4f& a, const Quat4f& b );

    union
    {
        struct
        {
            float w;
            float x;
            float y;
            float z;
        };
        float m_elements[ 4 ];
    };

};

Quat4f operator + ( const Quat4f& q0, const Quat4f& q1 );
Quat4f operator - ( const Quat4f& q0, const Quat4f& q1 );
Quat4f operator - ( const Quat4f& q );
Quat4f operator * ( const Quat4f& q0, const Quat4f& q1 );
Quat4f operator * ( float f, const Quat4f& q );
Quat4f operator * ( const Quat4f& q, float f );
