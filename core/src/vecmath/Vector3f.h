#pragma once

class QString;

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

    // no destructor necessary

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

    void homogenize();
    Vector3f homogenized() const;

    void negate();

    // automatic type conversion to float pointer
    operator const float* () const;
    operator float* ();
    QString toString() const;

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

Vector3f operator - ( const Vector3f& v0, const Vector3f& v1 );
// negate
Vector3f operator - ( const Vector3f& v );

Vector3f operator * ( float f, const Vector3f& v );
Vector3f operator * ( const Vector3f& v, float f );

// component-wise multiplication
Vector3f operator * ( const Vector3f& v0, const Vector3f& v1 );

// component-wise division
Vector3f operator / ( const Vector3f& v, float f );
Vector3f operator / ( const Vector3f& v0, const Vector3f& v1 );

// reciprocal of each component
Vector3f operator / ( float f, const Vector3f& v );

inline bool operator == ( const Vector3f& v0, const Vector3f& v1 );
inline bool operator != ( const Vector3f& v0, const Vector3f& v1 );

#include "Vector3f.inl"
