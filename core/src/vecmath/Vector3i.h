#pragma once

#include <string>

#include "Vector2i.h"

class Vector3f;

class Vector3i
{
public:

    Vector3i();
    explicit Vector3i( int i ); // fills all 3 components with i
    Vector3i( int x, int y, int z );
    Vector3i( const Vector2i& xy, int z );
    Vector3i( int x, const Vector2i& yz );

    // returns the ith element
    const int& operator [] ( int i ) const;
    int& operator [] ( int i );

    Vector2i zx() const;

    Vector2i yx() const;
    Vector2i zy() const;
    Vector2i xz() const;

    // TODO: all the other combinations

    Vector3i xyz() const;
    Vector3i yzx() const;
    Vector3i zxy() const;
    // TODO: all the other combinations

    float norm() const;
    int normSquared() const;
    Vector3f normalized() const;

    // implicit cast
    operator const int* () const;
    operator int* ();
    std::string toString() const;

    static int dot( const Vector3i& v0, const Vector3i& v1 );

    static Vector3i cross( const Vector3i& v0, const Vector3i& v1 );

    union
    {
        struct
        {
            int x;
            int y;
            int z;
        };
        struct
        {
            Vector2i xy;
        };
        struct
        {
            int __padding0;
            Vector2i yz;
        };
        int m_elements[ 3 ];
    };

};

bool operator == ( const Vector3i& v0, const Vector3i& v1 );
bool operator != ( const Vector3i& v0, const Vector3i& v1 );

Vector3i operator + ( const Vector3i& v0, const Vector3i& v1 );
Vector3i operator - ( const Vector3i& v0, const Vector3i& v1 );
Vector3i operator * ( const Vector3i& v0, const Vector3i& v1 );
Vector3i operator / ( const Vector3i& v0, const Vector3i& v1 );

Vector3i operator - ( const Vector3i& v );
Vector3i operator * ( int c, const Vector3i& v );
Vector3i operator * ( const Vector3i& v, int c );

Vector3f operator * ( float f, const Vector3i& v );
Vector3f operator * ( const Vector3i& v, float f );

Vector3i operator / ( const Vector3i& v, int c );
Vector3i operator / ( const Vector3i& v0, const Vector3i& v1 );
