#include "vecmath/Rect2i.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

Rect2i::Rect2i( int originX, int originY, int width, int height ) :

	m_origin( originX, originY ),
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

Vector2i Rect2i::size() const
{
	return m_size;
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
