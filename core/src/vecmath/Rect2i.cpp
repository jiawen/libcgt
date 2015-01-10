#include "vecmath/Rect2i.h"

#include <QString>

#include "math/MathUtils.h"
#include "vecmath/Vector2f.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

Rect2i::Rect2i() :
    m_origin( 0 ),
    m_size( 0 )
{

}

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

Rect2i::Rect2i( std::initializer_list< int > os )
{
    m_origin.x = *( os.begin() );
    m_origin.y = *( os.begin() + 1 );
    m_size.x = *( os.begin() + 2 );
    m_size.y = *( os.begin() + 3 );
}

Rect2i::Rect2i( const Rect2i& copy ) :
	m_origin( copy.m_origin ),
	m_size( copy.m_size )
{

}

Rect2i& Rect2i::operator = ( const Rect2i& copy )
{
	if( this != &copy )
	{
		m_origin = copy.m_origin;
		m_size = copy.m_size;
	}
	return *this;
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

int Rect2i::left() const
{
	return std::min( m_origin.x, m_origin.x + m_size.x );
}

int Rect2i::right() const
{
	return left() + width();
}

int Rect2i::bottom() const
{
	return std::min( m_origin.y, m_origin.y + m_size.y );
}

int Rect2i::top() const
{
	return bottom() + height();
}

Vector2i Rect2i::bottomLeft() const
{
    return{ left(), bottom() };
}

Vector2i Rect2i::bottomRight() const
{
    return{ right(), bottom() };
}

Vector2i Rect2i::topLeft() const
{
    return{ left(), top() };
}

Vector2i Rect2i::topRight() const
{
    return{ right(), top() };
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

Vector2f Rect2i::center() const
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

QString Rect2i::toString() const
{
	QString out;

	out.append( "Rect2f:\n" );
	out.append( "\torigin: " );
	out.append( m_origin.toString() );
	out.append( "\n\tsize: " );
	out.append( m_size.toString() );

	return out;
}

bool Rect2i::contains( const Vector2i& p )
{
	return
	(
		( p.x >= left() ) &&
		( p.x < right() ) &&
        ( p.y >= bottom() ) &&
		( p.y < top() )
	);
}

Rect2i Rect2i::flippedUD( int height ) const
{
	Vector2i origin;
	origin.x = m_origin.x;
	origin.y = height - topLeft().y;

	return Rect2i( origin, m_size );
}

// static
Rect2i Rect2i::united( const Rect2i& r0, const Rect2i& r1 )
{
    Vector2i unitedMin = MathUtils::minimum( r0.bottomLeft(), r1.bottomLeft() );
	Vector2i unitedMax = MathUtils::maximum( r0.topRight(), r1.topRight() );

	return Rect2i( unitedMin, unitedMax - unitedMin );
}
