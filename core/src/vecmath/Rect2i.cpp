#include "vecmath/Rect2i.h"

#include <cassert>

#include "math/MathUtils.h"
#include "vecmath/Rect2f.h"
#include "vecmath/Vector2f.h"

Rect2i::Rect2i( const Vector2i& size ) :
    origin( 0 ),
    size( size )
{

}

Rect2i::Rect2i( const Vector2i& origin, const Vector2i& size ) :
    origin( origin ),
    size( size )
{

}

int Rect2i::width() const
{
    return size.x;
}

int Rect2i::height() const
{
    return size.y;
}

int Rect2i::area() const
{
    return size.x * size.y;
}

int Rect2i::left() const
{
    return origin.x;
}

int Rect2i::right() const
{
    return origin.x + size.x;
}

int Rect2i::bottom() const
{
    return origin.y;
}

int Rect2i::top() const
{
    return origin.y + size.y;
}

Vector2i Rect2i::leftBottom() const
{
    return{ left(), bottom() };
}

Vector2i Rect2i::rightBottom() const
{
    return{ right(), bottom() };
}

Vector2i Rect2i::leftTop() const
{
    return{ left(), top() };
}

Vector2i Rect2i::rightTop() const
{
    return{ right(), top() };
}

Vector2i Rect2i::dx() const
{
    return{ size.x, 0 };
}

Vector2i Rect2i::dy() const
{
    return{ 0, size.y };
}

Vector2i Rect2i::minimum() const
{
    return libcgt::core::math::minimum( origin, origin + size );
}

Vector2i Rect2i::maximum() const
{
    return libcgt::core::math::maximum( origin, origin + size );
}

Vector2i Rect2i::center() const
{
    return ( origin + size ) / 2;
}

Vector2f Rect2i::exactCenter() const
{
    return origin + 0.5f * size;
}

bool Rect2i::isEmpty() const
{
    return( size.x == 0 || size.y == 0 );
}

bool Rect2i::isStandard() const
{
    return( size.x >= 0 && size.y >= 0 );
}

Rect2i Rect2i::standardized() const
{
    Vector2i origin2;
    Vector2i size2;

    if( size.x > 0 )
    {
        origin2.x = origin.x;
        size2.x = size.x;
    }
    else
    {
        origin2.x = origin.x + size.x;
        size2.x = -size.x;
    }

    if( size.y > 0 )
    {
        origin2.y = origin.y;
        size2.y = size.y;
    }
    else
    {
        origin2.y = origin.y + size.y;
        size2.y = -size.y;
    }

    return{ origin2, size2 };
}

std::string Rect2i::toString() const
{
    std::string out;

    out.append( "Rect2f:\n" );
    out.append( "\torigin: " );
    out.append( origin.toString() );
    out.append( "\n\tsize: " );
    out.append( size.toString() );

    return out;
}

bool Rect2i::contains( const Vector2i& p )
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

Rect2f Rect2i::castToFloat() const
{
    return Rect2f( origin, size );
}

// static
Rect2i Rect2i::united( const Rect2i& r0, const Rect2i& r1 )
{
    Vector2i r0Min = r0.minimum();
    Vector2i r0Max = r0.maximum();
    Vector2i r1Min = r1.minimum();
    Vector2i r1Max = r1.maximum();

    Vector2i unitedMin{ std::min( r0Min.x, r1Min.x ), std::min( r0Min.y, r1Min.y ) };
    Vector2i unitedMax{ std::max( r0Max.x, r1Max.x ), std::max( r0Max.y, r1Max.y ) };

    return Rect2i( unitedMin, unitedMax - unitedMin );
}
