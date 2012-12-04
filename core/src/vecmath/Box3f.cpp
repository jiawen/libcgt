#include "vecmath/Box3f.h"

#include <QString>

#include "math/Arithmetic.h"
#include "math/MathUtils.h"
#include "vecmath/Box3i.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

Box3f::Box3f() :

	m_origin( 0.f, 0.f, 0.f ),
	m_size( 0.f, 0.f, 0.f )

{

}

Box3f::Box3f( float left, float bottom, float back, float width, float height, float depth ) :

	m_origin( left, bottom, back ),
	m_size( width, height, depth )

{

}

Box3f::Box3f( float width, float height, float depth ) :

	m_origin( 0.f, 0.f, 0.f ),
	m_size( width, height, depth )

{

}

Box3f::Box3f( const Vector3f& origin, const Vector3f& size ) :

	m_origin( origin ),
	m_size( size )

{

}

Box3f::Box3f( const Vector3f& size ) :

	m_origin( 0.f, 0.f, 0.f ),
	m_size( size )

{

}

Box3f::Box3f( const Box3f& copy ) :

	m_origin( copy.m_origin ),
	m_size( copy.m_size )

{

}

Box3f& Box3f::operator = ( const Box3f& copy )
{
	if( this != &copy )
	{
		m_origin = copy.m_origin;
		m_size = copy.m_size;
	}
	return *this;
}

Vector3f Box3f::origin() const
{
	return m_origin;
}

Vector3f& Box3f::origin()
{
	return m_origin;
}

Vector3f Box3f::size() const
{
	return m_size;
}

Vector3f& Box3f::size()
{
	return m_size;
}

float Box3f::left() const
{
	return m_origin.x;
}

float Box3f::right() const
{
	return left() + width();
}

float Box3f::bottom() const
{
	return m_origin.y;
}

float Box3f::top() const
{
	return bottom() + height();
}

float Box3f::back() const
{
	return m_origin.z;
}

float Box3f::front() const
{
	return back() + depth();
}

Vector3f Box3f::leftBottomBack() const
{
	return m_origin;
}

Vector3f Box3f::rightBottomBack() const
{
	return m_origin + Vector3f( m_size.x, 0, 0 );
}

Vector3f Box3f::leftTopBack() const
{
	return m_origin + Vector3f( 0, m_size.y, 0 );
}

Vector3f Box3f::rightTopBack() const
{
	return m_origin + Vector3f( m_size.x, m_size.y, 0 );
}

Vector3f Box3f::leftBottomFront() const
{
	return m_origin + Vector3f( 0, 0, m_size.z );
}

Vector3f Box3f::rightBottomFront() const
{
	return m_origin + Vector3f( m_size.x, 0, m_size.z );
}

Vector3f Box3f::leftTopFront() const
{
	return m_origin + Vector3f( 0, m_size.y, m_size.z );
}

Vector3f Box3f::rightTopFront() const
{
	return m_origin + Vector3f( m_size.x, m_size.y, m_size.z );
}

float Box3f::width() const
{
	return m_size.x;
}

float Box3f::height() const
{
	return m_size.y;
}

float Box3f::depth() const
{
	return m_size.z;
}

float Box3f::volume() const
{
	return( m_size.x * m_size.y * m_size.z );
}

Vector3f Box3f::center() const
{
	return m_origin + 0.5f * m_size;
}

bool Box3f::isNull() const
{
	return( m_size.x == 0 && m_size.y == 0 );
}

bool Box3f::isEmpty() const
{
	return( !isValid() );
}

bool Box3f::isValid() const
{
	return( m_size.x > 0 && m_size.y > 0 );
}

Box3f Box3f::normalized() const
{
	Vector3f origin;
	Vector3f size;

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

	return Box3f( origin, size );
}

QString Box3f::toString() const
{
	QString out;

	out.append( "Box3f:\n" );
	out.append( "\torigin: " );
	out.append( m_origin.toString() );
	out.append( "\n\tsize: " );
	out.append( m_size.toString() );

	return out;
}

Box3f Box3f::flippedUD( float height ) const
{
	Vector3f origin;
	origin.x = m_origin.x;
	origin.y = height - top();
	origin.z = m_origin.z;

	return Box3f( origin, m_size );
}

Box3f Box3f::flippedBF( float depth ) const
{
	Vector3f origin;
	origin.x = m_origin.x;
	origin.y = m_origin.y;
	origin.z = depth - front();

	return Box3f( origin, m_size );
}

Box3f Box3f::flippedUDBF( float height, float depth ) const
{
	Vector3f origin;
	origin.x = m_origin.x;
	origin.y = height - top();
	origin.z = depth - front();

	return Box3f( origin, m_size );
}

Box3i Box3f::enlargedToInt() const
{
	Vector3i minimum = Arithmetic::floorToInt( leftBottomBack() );
	Vector3i maximum = Arithmetic::ceilToInt( rightTopFront() );

	// size does not need a +1:
	// say min is 1.1 and max is 3.6
	// then floor( min ) = 1 and ceil( 3.6 ) is 4
	// hence, we want indices 1, 2, 3, which has size 3 = 4 - 1
	Vector3i size = maximum - minimum;

	return Box3i( minimum, size );
}

bool Box3f::contains( float x, float y, float z )
{
	return
	(
		( x >= m_origin.x ) &&
		( x < ( m_origin.x + m_size.x ) ) &&
		( y >= m_origin.y ) &&
		( y < ( m_origin.y + m_size.y ) ) &&
		( z >= m_origin.z ) &&
		( z < ( m_origin.z + m_size.z ) )
	);
}

bool Box3f::contains( const Vector3f& p )
{
	return contains( p.x, p.y, p.z );
}

void Box3f::enlargeToContain( const Vector3f& p )
{
	m_origin = MathUtils::minimum( m_origin, p );
	Vector3f rtf = MathUtils::maximum( rightTopFront(), p );

	m_size = rtf - m_origin;	
}

// static
Box3f Box3f::united( const Box3f& b0, const Box3f& b1 )
{
	Vector3f unitedMin = MathUtils::minimum( b0.leftBottomBack(), b1.leftBottomBack() );
	Vector3f unitedMax = MathUtils::maximum( b0.rightTopFront(), b1.rightTopFront() );

	return Box3f( unitedMin, unitedMax - unitedMin );
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
	Vector3f minimum = MathUtils::maximum( b0.leftBottomBack(), b1.leftBottomBack() );
	Vector3f maximum = MathUtils::minimum( b0.rightTopFront(), b1.rightTopFront() );

	if( minimum.x < maximum.x &&
		minimum.y < maximum.y &&
		minimum.z < maximum.z )
	{
		intersection.m_origin = minimum;
		intersection.m_size = maximum - minimum;
		return true;
	}
	return false;
}

/*
std::vector< Vector3f > BoundingBox3f::corners() const
{
std::vector< Vector3f > out( 8 );

for( int i = 0; i < 8; ++i )
{
out[ i ] =
Vector3f
(
( i & 1 ) ? minimum().x : maximum().x,
( i & 2 ) ? minimum().y : maximum().y,
( i & 4 ) ? minimum().z : maximum().z
);
}

return out;
}

void BoundingBox3f::scale( const Vector3f& s )
{
Vector3f c = center();
Vector3f r = range();	
Vector3f r2 = s * r;

minimum() = c - 0.5f * r2;
maximum() = minimum() + r2;
}
*/