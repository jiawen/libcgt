#include "vecmath/Box3f.h"

#include "math/Arithmetic.h"
#include "math/MathUtils.h"
#include "vecmath/Box3i.h"

using libcgt::core::math::ceilToInt;
using libcgt::core::math::floorToInt;

Box3f::Box3f( const Vector3f& size ) :
    size{ size }
{

}

Box3f::Box3f( const Vector3f& origin, const Vector3f& size ) :
    origin{ origin },
    size{ size }
{

}

Vector3f Box3f::minimum() const
{
    return libcgt::core::math::minimum( origin, origin + size );
}

Vector3f Box3f::maximum() const
{
    return libcgt::core::math::maximum( origin, origin + size );
}

float Box3f::left() const
{
    return origin.x;
}

float Box3f::right() const
{
    return left() + width();
}

float Box3f::bottom() const
{
    return origin.y;
}

float Box3f::top() const
{
    return bottom() + height();
}

float Box3f::back() const
{
    return origin.z;
}

float Box3f::front() const
{
    return back() + depth();
}

Vector3f Box3f::leftBottomBack() const
{
    return origin;
}

Vector3f Box3f::rightBottomBack() const
{
    return origin + Vector3f( size.x, 0, 0 );
}

Vector3f Box3f::leftTopBack() const
{
    return origin + Vector3f( 0, size.y, 0 );
}

Vector3f Box3f::rightTopBack() const
{
    return origin + Vector3f( size.x, size.y, 0 );
}

Vector3f Box3f::leftBottomFront() const
{
    return origin + Vector3f( 0, 0, size.z );
}

Vector3f Box3f::rightBottomFront() const
{
    return origin + Vector3f( size.x, 0, size.z );
}

Vector3f Box3f::leftTopFront() const
{
    return origin + Vector3f( 0, size.y, size.z );
}

Vector3f Box3f::rightTopFront() const
{
    return origin + Vector3f( size.x, size.y, size.z );
}

float Box3f::width() const
{
    return size.x;
}

float Box3f::height() const
{
    return size.y;
}

float Box3f::depth() const
{
    return size.z;
}

float Box3f::volume() const
{
    return( size.x * size.y * size.z );
}

Vector3f Box3f::center() const
{
    return origin + 0.5f * size;
}

bool Box3f::isEmpty() const
{
    return( size.x == 0 || size.y == 0 || size.z == 0 );
}

bool Box3f::isStandard() const
{
    return( size.x >= 0 && size.y >= 0 && size.z >= 0 );
}

Box3f Box3f::standardized() const
{
    Vector3f origin2;
    Vector3f size2;

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

std::string Box3f::toString() const
{
    std::string out;

    out.append( "Box3f:\n" );
    out.append( "\torigin: " );
    out.append( origin.toString() );
    out.append( "\n\tsize: " );
    out.append( size.toString() );

    return out;
}

Box3f Box3f::flippedLR( float width ) const
{
    Vector3f origin;
    origin.x = width - left();
    origin.y = origin.y;
    origin.z = origin.z;

    return{ origin, size };
}

Box3f Box3f::flippedUD( float height ) const
{
    Vector3f origin;
    origin.x = origin.x;
    origin.y = height - top();
    origin.z = origin.z;

    return{ origin, size };
}

Box3f Box3f::flippedBF( float depth ) const
{
    Vector3f origin;
    origin.x = origin.x;
    origin.y = origin.y;
    origin.z = depth - front();

    return{ origin, size };
}

Box3i Box3f::enlargedToInt() const
{
    Vector3i minimum = floorToInt( leftBottomBack() );
    Vector3i maximum = ceilToInt( rightTopFront() );

    // size does not need a +1:
    // say min is 1.1 and max is 3.6
    // then floor( min ) = 1 and ceil( 3.6 ) is 4
    // hence, we want indices 1, 2, 3, which has size 3 = 4 - 1
    Vector3i size = maximum - minimum;

    return Box3i( minimum, size );
}

bool Box3f::contains( const Vector3f& p )
{
    return
    (
        ( p.x >= origin.x ) &&
        ( p.x < ( origin.x + size.x ) ) &&
        ( p.y >= origin.y ) &&
        ( p.y < ( origin.y + size.y ) ) &&
        ( p.z >= origin.z ) &&
        ( p.z < ( origin.z + size.z ) )
    );
}

void Box3f::enlargeToContain( const Vector3f& p )
{
    origin = libcgt::core::math::minimum( origin, p );
    Vector3f rtf = libcgt::core::math::maximum( rightTopFront(), p );

    size = rtf - origin;
}

// static
Box3f Box3f::united( const Box3f& b0, const Box3f& b1 )
{
    Vector3f unitedMin = libcgt::core::math::minimum(
        b0.leftBottomBack(), b1.leftBottomBack() );
    Vector3f unitedMax = libcgt::core::math::maximum(
        b0.rightTopFront(), b1.rightTopFront() );

    return{ unitedMin, unitedMax - unitedMin };
}

// static
bool Box3f::intersect( const Box3f& b0, const Box3f& b1 )
{
    Box3f isect;
    return intersect( b0, b1, isect );
}

// static
bool Box3f::intersect( const Box3f& b0, const Box3f& b1, Box3f& intersection )
{
    Vector3f minimum = libcgt::core::math::maximum(
        b0.leftBottomBack(), b1.leftBottomBack() );
    Vector3f maximum = libcgt::core::math::minimum(
        b0.rightTopFront(), b1.rightTopFront() );

    if( minimum.x < maximum.x &&
        minimum.y < maximum.y &&
        minimum.z < maximum.z )
    {
        intersection.origin = minimum;
        intersection.size = maximum - minimum;
        return true;
    }
    return false;
}

bool intersectRay( const Box3f& box,
    const Vector3f& rayOrigin, const Vector3f& rayDirection,
    float& tIntersect, float tMin )
{
    assert( box.isStandard() );
    assert( !box.isEmpty() );

    float tNear;
    float tFar;
    bool intersect = intersectLine( box, rayOrigin, rayDirection,
        tNear, tFar );
    if( intersect )
    {
        if( tNear >= tMin )
        {
            tIntersect = tNear;
        }
        else if( tFar >= tMin )
        {
            tIntersect = tFar;
        }
        else
        {
            intersect = false;
        }
    }
    return intersect;
}

bool intersectLine( const Box3f& box,
    const Vector3f& rayOrigin, const Vector3f& rayDirection,
    float& tNear, float& tFar )
{
    assert( box.isStandard() );
    assert( !box.isEmpty() );

    // Compute t to each face.
    Vector3f rcpDir = 1.0f / rayDirection;

    // Three "bottom" faces (min of the box).
    Vector3f tBottom = rcpDir * ( box.origin - rayOrigin );
    // three "top" faces (max of the box)
    Vector3f tTop = rcpDir * ( box.rightTopFront() - rayOrigin );

    // find the smallest and largest distances along each axis
    Vector3f tMin = libcgt::core::math::minimum( tBottom, tTop );
    Vector3f tMax = libcgt::core::math::maximum( tBottom, tTop );

    // tNear is the largest tMin
    tNear = libcgt::core::math::maximum( tMin );

    // tFar is the smallest tMax
    tFar = libcgt::core::math::minimum( tMax );

    return tFar > tNear;
}

bool carefulIntersectBoxRay( const Box3f& box,
    const Vector3f& rayOrigin, const Vector3f& rayDirection,
    float& t0, float& t1, int& t0Face, int& t1Face,
    float rayTMin, float rayTMax )
{
    assert( box.isStandard() );
    assert( !box.isEmpty() );

    t0 = rayTMin;
    t1 = rayTMax;
    t0Face = -1;
    t1Face = -1;

    // Compute t to each face.
    Vector3f rcpDir = 1.0f / rayDirection;

    Vector3f boxMax = box.rightTopFront();

    for( int i = 0; i < 3; ++i )
    {
        // Compute the intersection between the line and the slabs along the
        // i-th axis, parameterized as [tNear, tFar].
        float rcpDir = 1.0f / rayDirection[ i ];
        float tNear = rcpDir * ( box.origin[ i ] - rayOrigin[ i ] );
        float tFar = rcpDir * ( boxMax[ i ] - rayOrigin[ i ] );

        // Which face we're testing against.
        int nearFace = 2 * i;
        int farFace = 2 * i + 1;

        // Swap such that tNear < tFAr.
        if( tNear > tFar )
        {
            std::swap( tNear, tFar );
            std::swap( nearFace, farFace );
        }

        // Compute the set intersection between [tNear, tFar] and [t0, t1].
        if( tNear > t0 )
        {
            t0 = tNear;
            t0Face = nearFace;
        }
        if( tFar < t1 )
        {
            t1 = tFar;
            t1Face = farFace;
        }

        // Early abort if the range is empty.
        if( t0 > t1 )
        {
            return false;
        }
    }

    return true;
}

// static
Box3f Box3f::scale( const Box3f& b, const Vector3f& s )
{
    Vector3f size = s * b.size;
    return{ b.center() - 0.5f * size, size };
}
