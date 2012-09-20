#include "vecmath/Rect2i.h"

#include <QString>

#include "vecmath/Vector2f.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

Rect2i::Rect2i() :

	m_origin( 0, 0 ),
	m_size( 0, 0 )

{

}

Rect2i::Rect2i( int originX, int originY, int width, int height ) :

	m_origin( originX, originY ),
	m_size( width, height )

{

}

Rect2i::Rect2i( int width, int height ) :

	m_origin( 0, 0 ),
	m_size( width, height )

{

}

Rect2i::Rect2i( const Vector2i& origin, const Vector2i& size ) :

	m_origin( origin ),
	m_size( size )

{

}

Rect2i::Rect2i( const Vector2i& size ) :

	m_origin( 0, 0 ),
	m_size( size )

{

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

Vector2i Rect2i::bottomLeft() const
{
	return m_origin;
}

Vector2i Rect2i::bottomRight() const
{
	return m_origin + Vector2i( m_size.x, 0 );
}

Vector2i Rect2i::topLeft() const
{
	return m_origin + Vector2i( 0, m_size.y );
}

Vector2i Rect2i::topRight() const
{
	return m_origin + m_size;
}

int Rect2i::width() const
{
	return m_size.x;
}

int Rect2i::height() const
{
	return m_size.y;
}

int Rect2i::area() const
{
	return( m_size.x * m_size.y );
}

Vector2f Rect2i::center() const
{
	return m_origin + 0.5f * m_size;
}

bool Rect2i::isNull() const
{
	return( m_size.x == 0 && m_size.y == 0 );
}

bool Rect2i::isEmpty() const
{
	return( !isValid() );
}

bool Rect2i::isValid() const
{
	return( m_size.x > 0 && m_size.y > 0 );
}

Rect2i Rect2i::normalized() const
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

bool Rect2i::contains( int x, int y )
{
	return
	(
		( x > m_origin.x ) &&
		( x < ( m_origin.x + m_size.x ) ) &&
		( y > m_origin.y ) &&
		( y < ( m_origin.y + m_size.y ) )
	);
}

bool Rect2i::contains( const Vector2i& p )
{
	return contains( p.x, p.y );
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
	Vector2i r0Min = r0.bottomLeft();
	Vector2i r0Max = r0.topRight();
	Vector2i r1Min = r1.bottomLeft();
	Vector2i r1Max = r1.topRight();

	Vector2i unitedMin( std::min( r0Min.x, r1Min.x ), std::min( r0Min.y, r1Min.y ) );
	Vector2i unitedMax( std::max( r0Max.x, r1Max.x ), std::max( r0Max.y, r1Max.y ) );

	return Rect2i( unitedMin, unitedMax - unitedMin );
}
