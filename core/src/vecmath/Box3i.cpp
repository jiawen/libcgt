#include "vecmath/Box3i.h"

#include <QString>

#include "math/MathUtils.h"
#include "vecmath/Vector3f.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

Box3i::Box3i() :

	m_origin( 0, 0, 0 ),
	m_size( 0, 0, 0 )

{

}

Box3i::Box3i( int originX, int originY, int originZ, int width, int height, int depth ) :

	m_origin( originX, originY, originZ ),
	m_size( width, height, depth )

{

}

Box3i::Box3i( int width, int height, int depth ) :

	m_origin( 0, 0, 0 ),
	m_size( width, height, depth )

{

}

Box3i::Box3i( const Vector3i& origin, const Vector3i& size ) :

	m_origin( origin ),
	m_size( size )

{

}

Box3i::Box3i( const Vector3i& size ) :

	m_origin( 0, 0, 0 ),
	m_size( size )

{

}

Box3i::Box3i( const Box3i& copy ) :

	m_origin( copy.m_origin ),
	m_size( copy.m_size )

{

}

Box3i& Box3i::operator = ( const Box3i& copy )
{
	if( this != &copy )
	{
		m_origin = copy.m_origin;
		m_size = copy.m_size;
	}
	return *this;
}

Vector3i Box3i::origin() const
{
	return m_origin;
}

Vector3i& Box3i::origin()
{
	return m_origin;
}

Vector3i Box3i::size() const
{
	return m_size;
}

Vector3i& Box3i::size()
{
	return m_size;
}

int Box3i::left() const
{
	return m_origin.x;
}

int Box3i::right() const
{
	return left() + width();
}

int Box3i::bottom() const
{
	return m_origin.y;
}

int Box3i::top() const
{
	return bottom() + height();
}

int Box3i::back() const
{
	return m_origin.z;
}

int Box3i::front() const
{
	return back() + depth();
}

Vector3i Box3i::leftBottomBack() const
{
	return m_origin;
}

Vector3i Box3i::rightBottomBack() const
{
	return m_origin + Vector3i( m_size.x, 0, 0 );
}

Vector3i Box3i::leftTopBack() const
{
	return m_origin + Vector3i( 0, m_size.y, 0 );
}

Vector3i Box3i::rightTopBack() const
{
	return m_origin + Vector3i( m_size.x, m_size.y, 0 );
}

Vector3i Box3i::leftBottomFront() const
{
	return m_origin + Vector3i( 0, 0, m_size.z );
}

Vector3i Box3i::rightBottomFront() const
{
	return m_origin + Vector3i( m_size.x, 0, m_size.z );
}

Vector3i Box3i::leftTopFront() const
{
	return m_origin + Vector3i( 0, m_size.y, m_size.z );
}

Vector3i Box3i::rightTopFront() const
{
	return m_origin + Vector3i( m_size.x, m_size.y, m_size.z );
}

int Box3i::width() const
{
	return m_size.x;
}

int Box3i::height() const
{
	return m_size.y;
}

int Box3i::depth() const
{
	return m_size.z;
}

int Box3i::volume() const
{
	return( m_size.x * m_size.y * m_size.z );
}

Vector3f Box3i::center() const
{
	return m_origin + 0.5f * m_size;
}

bool Box3i::isNull() const
{
	return( m_size.x == 0 && m_size.y == 0 );
}

bool Box3i::isEmpty() const
{
	return( !isValid() );
}

bool Box3i::isValid() const
{
	return( m_size.x > 0 && m_size.y > 0 );
}

Box3i Box3i::normalized() const
{
	Vector3i origin;
	Vector3i size;

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

	if( m_size.z > 0 )
	{
		origin.z = m_origin.z;
		size.z = m_size.z;
	}
	else
	{
		origin.z = m_origin.z + m_size.z;
		size.z = -m_size.z;
	}

	return Box3i( origin, size );
}

QString Box3i::toString() const
{
	QString out;

	out.append( "Box3f:\n" );
	out.append( "\torigin: " );
	out.append( m_origin.toString() );
	out.append( "\n\tsize: " );
	out.append( m_size.toString() );

	return out;
}

bool Box3i::contains( int x, int y, int z )
{
	return
	(
		( x > m_origin.x ) &&
		( x < ( m_origin.x + m_size.x ) ) &&
		( y > m_origin.y ) &&
		( y < ( m_origin.y + m_size.y ) ) &&
		( z > m_origin.z ) &&
		( z < ( m_origin.z + m_size.z ) )
	);
}

bool Box3i::contains( const Vector3i& p )
{
	return contains( p.x, p.y, p.z );
}

Box3i Box3i::flippedUD( int height ) const
{
	Vector3i origin;
	origin.x = m_origin.x;
	origin.y = height - top();
	origin.z = m_origin.z;

	return Box3i( origin, m_size );
}

Box3i Box3i::flippedBF( int depth ) const
{
	Vector3i origin;
	origin.x = m_origin.x;
	origin.y = m_origin.y;
	origin.z = depth - front();

	return Box3i( origin, m_size );
}

Box3i Box3i::flippedUDBF( int height, int depth ) const
{
	Vector3i origin;
	origin.x = m_origin.x;
	origin.y = height - top();
	origin.z = depth - front();

	return Box3i( origin, m_size );
}

// static
Box3i Box3i::united( const Box3i& r0, const Box3i& r1 )
{
	Vector3i unitedMin = MathUtils::minimum( r0.leftBottomBack(), r1.leftBottomBack() );
	Vector3i unitedMax = MathUtils::maximum( r0.rightTopFront(), r1.rightTopFront() );
	
	return Box3i( unitedMin, unitedMax - unitedMin );
}
