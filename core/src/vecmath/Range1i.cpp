#include "vecmath/Range1i.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>

Range1i::Range1i( int size ) :
    size{ size }
{
}

Range1i::Range1i( int origin, int size ) :
    origin{ origin },
    size{ size }
{

}

int Range1i::width() const
{
    return size;
}

int Range1i::left() const
{
    return origin;
}

int Range1i::right() const
{
    return origin + size;
}

int Range1i::minimum() const
{
    return std::min( origin, origin + size );
}

int Range1i::maximum() const
{
    return std::max( origin, origin + size );
}

float Range1i::center() const
{
    return origin + 0.5f * size;
}

bool Range1i::isEmpty() const
{
    return( size == 0 );
}

bool Range1i::isStandard() const
{
    return( size >= 0 );
}

Range1i Range1i::standardized() const
{
    int origin;
    int size;

    if( size > 0 )
    {
        origin = origin;
        size = size;
    }
    else
    {
        origin = origin + size;
        size = -size;
    }

    return{ origin, size };
}

std::string Range1i::toString() const
{
    std::string out;

    out.append( "Range1i:\n" );
    out.append( "\torigin: " );
    out.append( std::to_string(origin) );
    out.append( "\n\tsize: " );
    out.append( std::to_string(size) );

    return out;
}

bool Range1i::contains( int x )
{
    assert( isStandard() );

    return
    (
        ( x >= left() ) &&
        ( x < right() )
    );
}

// static
Range1i Range1i::united( const Range1i& r0, const Range1i& r1 )
{
    assert( isStandard() );

    int r0Min = r0.left();
    int r0Max = r0.right();
    int r1Min = r1.left();
    int r1Max = r1.right();

    int unitedMin{ std::min( r0Min, r1Min ) };
    int unitedMax{ std::max( r0Max, r1Max ) };

    return{ unitedMin, unitedMax - unitedMin };
}
