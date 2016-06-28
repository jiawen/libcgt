#include <cassert>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>

#include "vecmath/Vector2f.h"
#include "vecmath/Vector2d.h"
#include "vecmath/Vector2i.h"
#include "vecmath/Vector3f.h"

Vector2f::Vector2f( const Vector2d& v ) :
    x( static_cast< float >( v.x ) ),
    y( static_cast< float >( v.y ) )
{

}

Vector2f::Vector2f( const Vector2i& v ) :
    x( static_cast< float >( v.x ) ),
    y( static_cast< float >( v.y ) )
{

}

Vector2f& Vector2f::operator = ( const Vector2d& v )
{
    x = static_cast< float >( v.x );
    y = static_cast< float >( v.y );

    return *this;
}

Vector2f& Vector2f::operator = ( const Vector2i& v )
{
    x = static_cast< float >( v.x );
    y = static_cast< float >( v.y );

    return *this;
}

// ---- Utility ----
std::string Vector2f::toString() const
{
    std::ostringstream sstream;
    sstream << "( " << x << ", " << y << ")";
    return sstream.str();
}


//static
Vector3f Vector2f::cross( const Vector2f& v0, const Vector2f& v1 )
{
    return Vector3f
        (
            0,
            0,
            v0.x * v1.y - v0.y * v1.x
        );
}
