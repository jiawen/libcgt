#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>

#include "vecmath/Vector3f.h"
#include "vecmath/Vector3d.h"
#include "vecmath/Vector3i.h"

// static
const Vector3f Vector3f::ZERO = Vector3f( 0, 0, 0 );

// static
const Vector3f Vector3f::UP = Vector3f( 0, 1, 0 );

// static
const Vector3f Vector3f::RIGHT = Vector3f( 1, 0, 0 );

// static
const Vector3f Vector3f::FORWARD = Vector3f( 0, 0, -1 );

Vector3f::Vector3f( const Vector2f& _xy, float _z )
{
    xy = _xy;
    z = _z;
}

Vector3f::Vector3f( float _x, const Vector2f& _yz )
{
    x = _x;
    yz = _yz;
}

Vector3f::Vector3f( const Vector3d& v )
{
    x = static_cast< float >( v.x );
    y = static_cast< float >( v.y );
    z = static_cast< float >( v.z );
}

Vector3f::Vector3f( const Vector3i& v )
{
    x = static_cast< float >( v.x );
    y = static_cast< float >( v.y );
    z = static_cast< float >( v.z );
}

Vector3f& Vector3f::operator = ( const Vector3d& v )
{
    x = static_cast< float >( v.x );
    y = static_cast< float >( v.y );
    z = static_cast< float >( v.z );

    return *this;
}

Vector3f& Vector3f::operator = ( const Vector3i& v )
{
    x = static_cast< float >( v.x );
    y = static_cast< float >( v.y );
    z = static_cast< float >( v.z );

    return *this;
}

Vector2f Vector3f::xz() const
{
    return{ x, z };
}

Vector3f Vector3f::xyz() const
{
    return Vector3f( x, y, z );
}

Vector3f Vector3f::yzx() const
{
    return Vector3f( y, z, x );
}

Vector3f Vector3f::zxy() const
{
    return Vector3f( z, x, y );
}

float Vector3f::norm() const
{
    return sqrt( normSquared() );
}

float Vector3f::normSquared() const
{
    return( x * x + y * y + z * z );
}

void Vector3f::normalize()
{
    float rcpNorm = 1.0f / norm();
    x *= rcpNorm;
    y *= rcpNorm;
    z *= rcpNorm;
}

Vector3f Vector3f::normalized() const
{
    float rcpNorm = 1.0f / norm();
    return Vector3f
    (
        x * rcpNorm,
        y * rcpNorm,
        z * rcpNorm
    );
}

void Vector3f::homogenize()
{
    if( z != 0 )
    {
        float rcpZ = 1.0f / z;
        x *= rcpZ;
        y *= rcpZ;
        z = 1;
    }
}

Vector3f Vector3f::homogenized() const
{
    if( z != 0 )
    {
        float rcpZ = 1.0f / z;
        return Vector3f( rcpZ * x, rcpZ * y, 1 );
    }
    else
    {
        return Vector3f( x, y, z );
    }
}

std::string Vector3f::toString() const
{
    std::ostringstream sstream;
    sstream << "( " << x << ", " << y << ", " << z << ")";
    return sstream.str();
}
