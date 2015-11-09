#include "vecmath/Rect2i.h"

#include <QString>

#include "math/MathUtils.h"
#include "vecmath/Rect2f.h"
#include "vecmath/Vector2f.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

Rect2i::Rect2i( const Vector2i& size ) :
    m_origin( 0 ),
    m_size( size )
{

}

Rect2i::Rect2i( const Vector2i& origin, const Vector2i& size ) :
    m_origin( origin ),
    m_size( size )
{

}

Rect2i::Rect2i( int originX, int originY, int sizeX, int sizeY ) :
    m_origin( originX, originY ),
    m_size( sizeX, sizeY )
{

}

Vector2i Rect2i::origin() const
{
    return m_origin;
}

Vector2i& Rect2i::origin()
{
    return m_origin;
}

Vector2i Rect2i::size() const
{
    return m_size;
}

Vector2i& Rect2i::size()
{
    return m_size;
}

Vector2i Rect2i::limit() const
{
    return m_origin + m_size;
}

Vector2i Rect2i::minimum() const
{
    return MathUtils::minimum( m_origin, m_origin + m_size );
}

Vector2i Rect2i::maximum() const
{
    return MathUtils::maximum( m_origin, m_origin + m_size );
}

Vector2i Rect2i::dx() const
{
    return{ m_size.x, 0 };
}

Vector2i Rect2i::dy() const
{
    return{ 0, m_size.y };
}

int Rect2i::width() const
{
    return std::abs( m_size.x );
}

int Rect2i::height() const
{
    return std::abs( m_size.y );
}

int Rect2i::area() const
{
    return std::abs( m_size.x * m_size.y );
}

Vector2i Rect2i::center() const
{
    return ( m_origin + m_size ) / 2;
}

Vector2f Rect2i::exactCenter() const
{
    return m_origin + 0.5f * m_size;
}

bool Rect2i::isStandardized() const
{
    return( m_size.x >= 0 && m_size.y >= 0 );
}

Rect2i Rect2i::standardized() const
{
    Vector2i origin;
    Vector2i size;

    if( m_size.x > 0 )
    {
        origin.x = m_origin.x;
        size.x = m_size.x;
    }
    else
    {
        origin.x = m_origin.x + m_size.x;
        size.x = -m_size.x;
    }

    if( m_size.y > 0 )
    {
        origin.y = m_origin.y;
        size.y = m_size.y;
    }
    else
    {
        origin.y = m_origin.y + m_size.y;
        size.y = -m_size.y;
    }

    return Rect2i( origin, size );
}

std::string Rect2i::toString() const
{
    std::string out;

    out.append( "Rect2f:\n" );
    out.append( "\torigin: " );
    out.append( m_origin.toString().toStdString() );
    out.append( "\n\tsize: " );
    out.append( m_size.toString().toStdString() );

    return out;
}

bool Rect2i::contains( const Vector2i& p )
{
    return
    (
        ( p.x >= m_origin.x ) &&
        ( p.x < ( m_origin.x + m_size.x ) ) &&
        ( p.y >= m_origin.y ) &&
        ( p.y < ( m_origin.y + m_size.y ) )
    );
}

Rect2f Rect2i::castToFloat() const
{
    return Rect2f( m_origin, m_size );
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
