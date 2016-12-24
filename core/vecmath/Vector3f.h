#pragma once

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

#include "Vector2f.h"

class Vector3d;
class Vector3i;

class Vector3f
{
public:

    static const Vector3f ZERO;
    static const Vector3f UP;
    static const Vector3f RIGHT;
    static const Vector3f FORWARD;

    // Default constructor initializes to all 0s.
    Vector3f();
    explicit Vector3f( float f ); // (f,f,f)
    Vector3f( float _x, float _y, float _z );

    Vector3f( const Vector2f& _xy, float _z );
    Vector3f( float _x, const Vector2f& _yz );

    // copy constructors
    Vector3f( const Vector3f& v ) = default;
    Vector3f( const Vector3d& v );
    Vector3f( const Vector3i& v );

    // assignment operators
    Vector3f& operator = ( const Vector3f& v ) = default;
    Vector3f& operator = ( const Vector3d& v );
    Vector3f& operator = ( const Vector3i& v );

    // returns the ith element
    const float& operator [] ( int i ) const;
    float& operator [] ( int i );

    Vector2f xz() const;
    // TODO: all the other combinations

    Vector3f xyz() const;
    Vector3f yzx() const;
    Vector3f zxy() const;
    // TODO: all the other combinations

    float norm() const;
    float normSquared() const;

    void normalize();
    Vector3f normalized() const;
    // Decompose a vector into its scalar norm and unit direction.
    Vector3f normalized( float& normOut ) const;

    void homogenize();
    Vector3f homogenized() const;

    // automatic type conversion to float pointer
    operator const float* () const;
    operator float* ();
    std::string toString() const;

    // dot product
    static float dot( const Vector3f& v0, const Vector3f& v1 );

    // cross product
    static Vector3f cross( const Vector3f& v0, const Vector3f& v1 );

    Vector3f& operator += ( const Vector3f& v );
    Vector3f& operator -= ( const Vector3f& v );
    Vector3f& operator *= ( float f );
    Vector3f& operator /= ( float f );

    union
    {
        // Individual element access.
        struct
        {
            float x;
            float y;
            float z;
        };
        // Vector2.
        struct
        {
            Vector2f xy;
        };
        struct
        {
            float __padding0;
            Vector2f yz;
        };
    };

};

Vector3f operator + ( const Vector3f& v0, const Vector3f& v1 );
Vector3f operator + ( const Vector3f& v, float f );
Vector3f operator + ( float f, const Vector3f& v );

Vector3f operator - ( const Vector3f& v0, const Vector3f& v1 );
Vector3f operator - ( const Vector3f& v, float f );
Vector3f operator - ( float f, const Vector3f& v );
// Negate.
Vector3f operator - ( const Vector3f& v );

Vector3f operator * ( float f, const Vector3f& v );
Vector3f operator * ( const Vector3f& v, float f );

// Component-wise multiplication.
Vector3f operator * ( const Vector3f& v0, const Vector3f& v1 );

// Component-wise division.
Vector3f operator / ( const Vector3f& v, float f );
Vector3f operator / ( const Vector3f& v0, const Vector3f& v1 );

// Reciprocal of each component.
Vector3f operator / ( float f, const Vector3f& v );

bool operator == ( const Vector3f& v0, const Vector3f& v1 );
bool operator != ( const Vector3f& v0, const Vector3f& v1 );

#include "Vector3d.h"
#include "Vector3i.h"
#include "Vector3f.inl"
