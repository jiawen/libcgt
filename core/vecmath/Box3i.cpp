#include "libcgt/core/vecmath/Box3i.h"

#include <cassert>

#include "libcgt/core/math/MathUtils.h"
#include "libcgt/core/vecmath/Vector3f.h"

Box3i::Box3i( const Vector3i& size ) :
    size{ size }
{

}

Box3i::Box3i( const Vector3i& origin, const Vector3i& size ) :
    origin( origin ),
    size( size )
{

}

int Box3i::width() const
{
    return size.x;
}

int Box3i::height() const
{
    return size.y;
}

int Box3i::depth() const
{
    return size.z;
}

int Box3i::volume() const
{
    return size.x * size.y * size.z;
}

int Box3i::left() const
{
    return origin.x;
}

int Box3i::right() const
{
    return origin.x + size.x;
}

int Box3i::bottom() const
{
    return origin.y;
}

int Box3i::top() const
{
    return origin.y + size.y;
}

int Box3i::back() const
{
    return origin.z;
}

int Box3i::front() const
{
    return origin.z + size.z;
}

Vector3i Box3i::leftBottomBack() const
{
    return{ left(), bottom(), back() };
}

Vector3i Box3i::rightBottomBack() const
{
    return{ right(), bottom(), back() };
}

Vector3i Box3i::leftTopBack() const
{
    return{ left(), top(), back() };
}

Vector3i Box3i::rightTopBack() const
{
    return{ right(), top(), back() };
}

Vector3i Box3i::leftBottomFront() const
{
    return{ left(), bottom(), front() };
}

Vector3i Box3i::rightBottomFront() const
{
    return{ right(), bottom(), front() };
}

Vector3i Box3i::leftTopFront() const
{
    return{ left(), top(), front() };
}

Vector3i Box3i::rightTopFront() const
{
    return{ right(), top(), front() };
}

Vector3i Box3i::minimum() const
{
    return libcgt::core::math::minimum( origin, origin + size );
}

Vector3i Box3i::maximum() const
{
    return libcgt::core::math::maximum(origin, origin + size);
}

Vector3f Box3i::center() const
{
    return origin + 0.5f * size;
}

bool Box3i::isEmpty() const
{
    return( size.x == 0 || size.y == 0 || size.z == 0 );
}

bool Box3i::isStandard() const
{
    return( size.x >= 0 && size.y >= 0 && size.z >= 0 );
}

Box3i Box3i::standardized() const
{
    Vector3i origin2;
    Vector3i size2;

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

    if( size.z > 0 )
    {
        origin2.z = origin.z;
        size2.z = size.z;
    }
    else
    {
        origin2.z = origin.z + size.z;
        size2.z = -size.z;
    }

    return{ origin, size };
}

std::string Box3i::toString() const
{
    std::string out;

    out.append( "Box3i:\n" );
    out.append( "\torigin: " );
    out.append( origin.toString() );
    out.append( "\n\tsize: " );
    out.append( size.toString() );

    return out;
}

bool Box3i::contains( const Vector3i& p )
{
    assert( isStandard() );

    return
    (
        ( p.x >= left() ) &&
        ( p.x < right() ) &&
        ( p.y >= bottom() ) &&
        ( p.y < top() ) &&
        ( p.z >= back() ) &&
        ( p.z < front() )
    );
}

Box3i Box3i::flippedUD( int height ) const
{
    assert( isStandard() );

    Vector3i origin;
    origin.x = origin.x;
    origin.y = height - top();
    origin.z = origin.z;

    return Box3i( origin, size );
}

Box3i Box3i::flippedBF( int depth ) const
{
    assert( isStandard() );

    Vector3i origin;
    origin.x = origin.x;
    origin.y = origin.y;
    origin.z = depth - front();

    return Box3i( origin, size );
}

// static
Box3i Box3i::united( const Box3i& r0, const Box3i& r1 )
{
    assert( r0.isStandard() && r1.isStandard() );

    Vector3i unitedMin = libcgt::core::math::minimum( r0.leftBottomBack(), r1.leftBottomBack() );
    Vector3i unitedMax = libcgt::core::math::maximum( r0.rightTopFront(), r1.rightTopFront() );

    return Box3i( unitedMin, unitedMax - unitedMin );
}
