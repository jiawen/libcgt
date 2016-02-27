#include "vecmath/Range1i.h"

#include <cmath>
#include <cstdlib>

using std::abs;

Range1i::Range1i() :
    m_origin( 0 ),
    m_size( 0 )
{

}

Range1i::Range1i( int size ) :
    m_origin( 0 ),
    m_size( size )
{
}

Range1i::Range1i( std::initializer_list< int > os )
{
    m_origin = *( os.begin() );
    m_size = *( os.begin() + 1 );
}

Range1i::Range1i( const Range1i& copy ) :

    m_origin( copy.m_origin ),
    m_size( copy.m_size )

{

}

Range1i& Range1i::operator = ( const Range1i& copy )
{
    if( this != &copy )
    {
        m_origin = copy.m_origin;
        m_size = copy.m_size;
    }
    return *this;
}

int Range1i::origin() const
{
    return m_origin;
}

int& Range1i::origin()
{
    return m_origin;
}

int Range1i::size() const
{
    return m_size;
}

int& Range1i::size()
{
    return m_size;
}

int Range1i::left() const
{
    return std::min( m_origin, m_origin + m_size );
}

int Range1i::right() const
{
    return left() + width();
}

int Range1i::width() const
{
    return abs( m_size );
}

float Range1i::center() const
{
    return m_origin + 0.5f * m_size;
}

bool Range1i::isStandardized() const
{
    return( m_size >= 0 );
}

Range1i Range1i::standardized() const
{
    int origin;
    int size;

    if( m_size > 0 )
    {
        origin = m_origin;
        size = m_size;
    }
    else
    {
        origin = m_origin + m_size;
        size = -m_size;
    }

    return{ origin, size };
}

std::string Range1i::toString() const
{
    std::string out;

    out.append( "Range1i:\n" );
    out.append( "\torigin: " );
    out.append( std::to_string(m_origin) );
    out.append( "\n\tsize: " );
    out.append( std::to_string(m_size) );

    return out;
}

bool Range1i::contains( int x )
{
    return
    (
        ( x >= left() ) &&
        ( x < right() )
    );
}

// static
Range1i Range1i::united( const Range1i& r0, const Range1i& r1 )
{
    int r0Min = r0.left();
    int r0Max = r0.right();
    int r1Min = r1.left();
    int r1Max = r1.right();

    int unitedMin{ std::min( r0Min, r1Min ) };
    int unitedMax{ std::max( r0Max, r1Max ) };

    return{ unitedMin, unitedMax - unitedMin };
}
