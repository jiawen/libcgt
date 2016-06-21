#include <cassert>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>

#include "vecmath/Vector2i.h"
#include "vecmath/Vector2f.h"
#include "vecmath/Vector3i.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

Vector2i::Vector2i( int i ) :
    x( i ),
    y( i )
{
}

Vector2i::Vector2i( int _x, int _y ) :
    x( _x ),
    y( _y )
{
}

const int& Vector2i::operator [] ( int i ) const
{
    return ( &x )[ i ];
}

int& Vector2i::operator [] ( int i )
{
    return ( &x )[ i ];
}

Vector2i Vector2i::xy() const
{
    return{ x, y };
}

Vector2i Vector2i::yx() const
{
    return{ y, x };
}

Vector2i Vector2i::xx() const
{
    return{ x, x };
}

Vector2i Vector2i::yy() const
{
    return{ y, y };
}

float Vector2i::norm() const
{
    return sqrt( static_cast< float >( normSquared() ) );
}

int Vector2i::normSquared() const
{
    return( x * x + y * y );
}

Vector2f Vector2i::normalized() const
{
    float n = 1.f / norm();

    return
    {
        n * x,
        n * y
    };
}

void Vector2i::negate()
{
    x = -x;
    y = -y;
}

Vector2i Vector2i::flippedUD( int height ) const
{
    return{ x, height - y - 1 };
}

Vector2i::operator const int* () const
{
    return &x;
}

Vector2i::operator int* ()
{
    return &x;
}

std::string Vector2i::toString() const
{
    std::ostringstream sstream;
    sstream << "( " << x << ", " << y << ")";
    return sstream.str();
}

// static
int Vector2i::dot( const Vector2i& v0, const Vector2i& v1 )
{
    return v0.x * v1.x + v0.y * v1.y;
}

//static
Vector3i Vector2i::cross( const Vector2i& v0, const Vector2i& v1 )
{
    return
    {
        0,
        0,
        v0.x * v1.y - v0.y * v1.x
    };
}

//////////////////////////////////////////////////////////////////////////
// Operators
//////////////////////////////////////////////////////////////////////////

Vector2i operator + ( const Vector2i& v0, const Vector2i& v1 )
{
    return{ v0.x + v1.x, v0.y + v1.y };
}

Vector2i operator - ( const Vector2i& v0, const Vector2i& v1 )
{
    return{ v0.x - v1.x, v0.y - v1.y };
}

Vector2i operator - ( const Vector2i& v )
{
    return{ -v.x, -v.y };
}

Vector2i operator * ( int c, const Vector2i& v )
{
    return{ c * v.x, c * v.y };
}

Vector2i operator * ( const Vector2i& v, int c )
{
    return{ c * v.x, c * v.y };
}

Vector2f operator * ( float f, const Vector2i& v )
{
    return{ f * v.x, f * v.y };
}

Vector2f operator * ( const Vector2i& v, float f )
{
    return{ f * v.x, f * v.y };
}

Vector2i operator * ( const Vector2i& v0, const Vector2i& v1 )
{
    return{ v0.x * v1.x, v0.y * v1.y };
}

Vector2i operator / ( const Vector2i& v0, const Vector2i& v1 )
{
    return{ v0.x / v1.x, v0.y / v1.y };
}

Vector2i operator / ( const Vector2i& v, int c )
{
    return{ v.x / c, v.y / c };
}

bool operator == ( const Vector2i& v0, const Vector2i& v1 )
{
    return
    (
        v0.x == v1.x &&
        v0.y == v1.y
    );
}

bool operator != ( const Vector2i& v0, const Vector2i& v1 )
{
    return !( v0 == v1 );
}
