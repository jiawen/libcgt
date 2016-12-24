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
    // axisAngle.normalized() is the direction. This function is numerically
    // robust near zero.
    //
    // This is equivalent to exp( axisAngle ).
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

    float norm() const;
    float normSquared() const;
    void normalize();
    Quat4f normalized() const;

    Quat4f conjugated() const;

    // Recall that the inverse of a quaternion q is conj(q) / norm(q).
    // If q is a unit quaternion, its inverse is its conjugate.
    Quat4f inverse() const;

    // TODO: deprecate this.
    Quat4f exp() const;

    // TODO: deprecate this.
    Quat4f log() const;

    // If this is a unit quaternion, returns the unit vector for rotation and
    // separately, the radians about the unit vector.
    //
    // The angle is guaranteed to be in [0, pi].
    // If this is the identity quaternion (1, 0, 0, 0), returns
    // axis = (1, 0, 0) and angle = 0.
    Vector3f getAxisAngle( float* radiansOut ) const;

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
    // Useful for squad().
    // TODO: deprecate and rewrite in terms of exp() and log().
    static Quat4f squadTangent( const Quat4f& before, const Quat4f& center,
        const Quat4f& after );

    // spherical cubic interpolation: given control points q[i] and parameter t
    // computes the interpolated quaternion
    static Quat4f cubicInterpolate(
        const Quat4f& q0, const Quat4f& q1, const Quat4f& q2, const Quat4f& q3,
        float t );

    union
    {
        // Individual element access.
        struct
        {
            float w;
            float x;
            float y;
            float z;
        };
        // Vector3
        struct
        {
            float __padding0;
            Vector3f xyz;
        };
        // Vector4
        struct
        {
            Vector4f wxyz;
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

// The exponential map from R^3 --> S^3 sends:
// (0,0,0)^T --> Quat(1, (0, 0, 0))
// v --> Quat(cos(theta/2), sin(theta/2) normalize(v))
//
// This function is numerically robust in the vicinity of 0.
Quat4f exp( const Vector3f& axisAngle );

// TODO: implement Jacobians. If R is a rotation matrix, q is its quaternion
// representation, and v its rotation vector representation:
//
// dR/dv = dR/dq dq/dv
//
// Note that dq/dv is a 4x3 matrix D[l, n], where l is the range over the
// quaternion and n is over the vector.
// You need to be careful with dq/dv numerically.

// The log map from S^3 --> R^3.
// v = 2 * acos(q.w) * normalize(v).
//
// This function is numerically robust in the vicinity of a rotation of 0 or
// 2pi: it will return the zero vector.
Vector3f log( const Quat4f& q );

// TODO: implement time derivatives. If v is a rotation vector, compute
// dv/dt as a function of dq/dt. dq/dt = 0.5f * Quat4f(0, omega) * q.
