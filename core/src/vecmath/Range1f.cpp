#include "vecmath/Range1f.h"

#define _USE_MATH_DEFINES
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>

#include "math/Arithmetic.h"
#include "vecmath/Range1i.h"

using libcgt::core::math::ceilToInt;
using libcgt::core::math::floorToInt;

Range1f::Range1f( float size ) :
    size{ size }
{

}

Range1f::Range1f( float origin, float size ) :
    origin{ origin },
    size{ size }
{

}

// static
Range1f Range1f::fromMinMax( float min, float max )
{
    return Range1f( min, max - min );
}

float Range1f::width() const
{
    return size;
}

float Range1f::left() const
{
    return origin;
}

float Range1f::right() const
{
    return origin + size;
}

float Range1f::minimum() const
{
    return std::min( origin, origin + size );
}

float Range1f::maximum() const
{
    return std::max( origin, origin + size );
}

float Range1f::center() const
{
    return origin + 0.5f * size;
}

bool Range1f::isEmpty() const
{
    return( size == 0 );
}

bool Range1f::isStandard() const
{
    return( size >= 0 );
}

Range1f Range1f::standardized() const
{
    if( size > 0 )
    {
        return{ origin, size };
    }
    else
    {
        return{ origin + size, -size };
    }
}

std::string Range1f::toString() const
{
    std::string out;

    out.append( "Range1f:\n" );
    out.append( "\torigin: " );
    out.append( std::to_string( origin ) );
    out.append( "\n\tsize: " );
    out.append( std::to_string( size ) );

    return out;
}

Range1i Range1f::enlargedToInt() const
{
    assert( isStandard() );

    int minimum = floorToInt( left() );
    int maximum = ceilToInt( right() );

    // size does not need a +1:
    // say min is 1.1 and max is 3.6
    // then floor( min ) = 1 and ceil( 3.6 ) is 4
    // hence, we want indices 1, 2, 3, which has size 3 = 4 - 1
    int size = maximum - minimum;

    return Range1i( minimum, size );
}

bool Range1f::contains( float x ) const
{
    assert( isStandard() );

    return
    (
        ( x >= left() ) &&
        ( x < right() )
    );
}

// static
Range1f Range1f::united( const Range1f& r0, const Range1f& r1 )
{
    assert( r0.isStandard() && r1.isStandard() );

    float r0Min = r0.left();
    float r0Max = r0.right();
    float r1Min = r1.left();
    float r1Max = r1.right();

    float unitedMin{ std::min( r0Min, r1Min ) };
    float unitedMax{ std::max( r0Max, r1Max ) };

    return fromMinMax( unitedMin, unitedMax );
}
