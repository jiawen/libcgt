#include "vecmath/Rect2f.h"

#include <cassert>

#include "math/Arithmetic.h"
#include "math/MathUtils.h"
#include "vecmath/Rect2i.h"

using libcgt::core::math::ceilToInt;
using libcgt::core::math::floorToInt;

Rect2f::Rect2f( const Vector2f& size ) :
    size( size )
{

}

Rect2f::Rect2f( const Vector2f& origin, const Vector2f& size ) :
    origin( origin ),
    size( size )
{

}

float Rect2f::width() const
{
    return size.x;
}

float Rect2f::height() const
{
    return size.y;
}

float Rect2f::area() const
{
    return( size.x * size.y );
}

float Rect2f::left() const
{
    return origin.x;
}

float Rect2f::right() const
{
    return origin.x + size.x;
}

float Rect2f::bottom() const
{
    return origin.y;
}

float Rect2f::top() const
{
    return origin.y + size.y;
}

Vector2f Rect2f::leftBottom() const
{
    return{ left(), bottom() };
}

Vector2f Rect2f::rightBottom() const
{
    return{ right(), bottom() };
}

Vector2f Rect2f::leftTop() const
{
    return{ left(), top() };
}

Vector2f Rect2f::rightTop() const
{
    return{ right(), top() };
}

Vector2f Rect2f::dx() const
{
    return{ size.x, 0 };
}

Vector2f Rect2f::dy() const
{
    return{ 0, size.y };
}

Vector2f Rect2f::minimum() const
{
    return libcgt::core::math::minimum( origin, origin + size );
}

Vector2f Rect2f::maximum() const
{
    return libcgt::core::math::maximum( origin, origin + size );
}

Vector2f Rect2f::center() const
{
    return origin + 0.5f * size;
}

bool Rect2f::isEmpty() const
{
    return( size.x == 0 || size.y == 0 );
}

bool Rect2f::isStandard() const
{
    return( size.x >= 0 && size.y >= 0 );
}

Rect2f Rect2f::standardized() const
{
    Vector2f origin;
    Vector2f size;

    if( size.x > 0 )
    {
        origin.x = origin.x;
        size.x = size.x;
    }
    else
    {
        origin.x = origin.x + size.x;
        size.x = -size.x;
    }

    if( size.y > 0 )
    {
        origin.y = origin.y;
        size.y = size.y;
    }
    else
    {
        origin.y = origin.y + size.y;
        size.y = -size.y;
    }

    return Rect2f( origin, size );
}

std::string Rect2f::toString() const
{
    std::string out;

    out.append( "Rect2f:\n" );
    out.append( "\torigin: " );
    out.append( origin.toString() );
    out.append( "\n\tsize: " );
    out.append( size.toString() );

    return out;
}

Rect2i Rect2f::enlargedToInt() const
{
    assert( isStandard() );

    Vector2i minimum = floorToInt( origin );
    Vector2i maximum = ceilToInt( rightTop() );

    // size does not need a +1:
    // say min is 1.1 and max is 3.6
    // then floor( min ) = 1 and ceil( 3.6 ) is 4
    // hence, we want indices 1, 2, 3, which has size 3 = 4 - 1
    Vector2i size = maximum - minimum;

    return Rect2i( minimum, size );
}

bool Rect2f::contains( const Vector2f& p ) const
{
    assert( isStandard() );

    return
    (
        ( p.x >= origin.x ) &&
        ( p.x < ( origin.x + size.x ) ) &&
        ( p.y >= origin.y ) &&
        ( p.y < ( origin.y + size.y ) )
    );
}

bool Rect2f::intersectRay( const Vector2f& rayOrigin, const Vector2f& rayDirection,
    float& tNear, float& tFar, int& axis ) const
{
    assert( isStandard() );

    // compute t to each face
    Vector2f rcpDir = 1.0f / rayDirection;

    // intersect left and bottom
    Vector2f tBottomLeft = rcpDir * ( leftBottom() - rayOrigin );
    // intersect right and top
    Vector2f tTopRight = rcpDir * ( rightTop() - rayOrigin );

    // find the smallest and largest distances along each axis
    Vector2f tMin = libcgt::core::math::minimum( tBottomLeft, tTopRight );
    Vector2f tMax = libcgt::core::math::maximum( tBottomLeft, tTopRight );

    // tNear is the largest tMin
    tNear = libcgt::core::math::maximum( tMin );

    // tFar is the smallest tMax
    tFar = libcgt::core::math::minimum( tMax );

    bool intersected = ( tFar > tNear );
    if( intersected )
    {
        if( tNear == tMin.x || tNear == tMax.x )
        {
            axis = 0;
        }
        else
        {
            axis = 1;
        }
    }

    return intersected;
}

// static
Rect2f Rect2f::united( const Rect2f& r0, const Rect2f& r1 )
{
    assert( r0.isStandard() && r1.isStandard() );

    Vector2f r0Min = r0.minimum();
    Vector2f r0Max = r0.maximum();
    Vector2f r1Min = r1.minimum();
    Vector2f r1Max = r1.maximum();

    Vector2f unitedMin{ std::min( r0Min.x, r1Min.x ), std::min( r0Min.y, r1Min.y ) };
    Vector2f unitedMax{ std::max( r0Max.x, r1Max.x ), std::max( r0Max.y, r1Max.y ) };

    return Rect2f( unitedMin, unitedMax - unitedMin );
}

// static
bool Rect2f::intersect( const Rect2f& r0, const Rect2f& r1 )
{
    Rect2f isect;
    return intersect( r0, r1, isect );
}

// static
bool Rect2f::intersect( const Rect2f& r0, const Rect2f& r1, Rect2f& intersection )
{
    assert( r0.isStandard() && r1.isStandard() );

    Vector2f minimum = libcgt::core::math::maximum( r0.minimum(), r1.minimum() );
    Vector2f maximum = libcgt::core::math::minimum( r0.maximum(), r1.maximum() );

    if( minimum.x < maximum.x &&
        minimum.y < maximum.y )
    {
        intersection.origin = minimum;
        intersection.size = maximum - minimum;
        return true;
    }
    return false;
}
