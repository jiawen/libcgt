#include "vecmath/Rect2f.h"

#include <QString>

#include "math/Arithmetic.h"
#include "math/MathUtils.h"
#include "vecmath/Rect2i.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

Rect2f::Rect2f() :

	m_origin( 0.f, 0.f ),
	m_size( 0.f, 0.f )

{

}

Rect2f::Rect2f( float originX, float originY, float width, float height ) :

	m_origin( originX, originY ),
	m_size( width, height )

{

}

Rect2f::Rect2f( float width, float height ) :

	m_origin( 0.f, 0.f ),
	m_size( width, height )

{

}

Rect2f::Rect2f( const Vector2f& origin, const Vector2f& size ) :

	m_origin( origin ),
	m_size( size )

{

}

Rect2f::Rect2f( const Vector2f& size ) :

	m_origin( 0.f, 0.f ),
	m_size( size )

{

}

Rect2f::Rect2f( const Rect2f& copy ) :

	m_origin( copy.m_origin ),
	m_size( copy.m_size )

{

}

Rect2f& Rect2f::operator = ( const Rect2f& copy )
{
	if( this != &copy )
	{
		m_origin = copy.m_origin;
		m_size = copy.m_size;
	}
	return *this;
}

Vector2f Rect2f::origin() const
{
	return m_origin;
}

Vector2f& Rect2f::origin()
{
	return m_origin;
}

Vector2f Rect2f::size() const
{
	return m_size;
}

Vector2f& Rect2f::size()
{
	return m_size;
}

float Rect2f::left() const
{
	return m_origin.x;
}

float Rect2f::right() const
{
	return left() + width();
}

float Rect2f::bottom() const
{
	return m_origin.y;
}

float Rect2f::top() const
{
	return bottom() + height();
}

Vector2f Rect2f::bottomLeft() const
{
	return m_origin;
}

Vector2f Rect2f::bottomRight() const
{
	return m_origin + Vector2f( m_size.x, 0.f );
}

Vector2f Rect2f::topLeft() const
{
	return m_origin + Vector2f( 0.f, m_size.y );
}

Vector2f Rect2f::topRight() const
{
	return m_origin + m_size;
}

float Rect2f::width() const
{
	return m_size.x;
}

float Rect2f::height() const
{
	return m_size.y;
}

float Rect2f::area() const
{
	return( m_size.x * m_size.y );
}

Vector2f Rect2f::center() const
{
	return m_origin + 0.5f * m_size;
}

bool Rect2f::isNull() const
{
	return( m_size.x == 0 && m_size.y == 0 );
}

bool Rect2f::isEmpty() const
{
	return( !isValid() );
}

bool Rect2f::isValid() const
{
	return( m_size.x > 0 && m_size.y > 0 );
}

Rect2f Rect2f::normalized() const
{
	Vector2f origin;
	Vector2f size;

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

	return Rect2f( origin, size );
}

QString Rect2f::toString() const
{
	QString out;

	out.append( "Rect2f:\n" );
	out.append( "\torigin: " );
	out.append( m_origin.toString() );
	out.append( "\n\tsize: " );
	out.append( m_size.toString() );

	return out;
}

Rect2f Rect2f::flippedUD( float height ) const
{
	Vector2f origin;
	origin.x = m_origin.x;
	origin.y = height - topLeft().y;

	return Rect2f( origin, m_size );
}

Rect2i Rect2f::enlargedToInt() const
{
	Vector2i minimum = Arithmetic::floorToInt( bottomLeft() );
	Vector2i maximum = Arithmetic::ceilToInt( topRight() );

	// size does not need a +1:
	// say min is 1.1 and max is 3.6
	// then floor( min ) = 1 and ceil( 3.6 ) is 4
	// hence, we want indices 1, 2, 3, which has size 3 = 4 - 1
	Vector2i size = maximum - minimum;

	return Rect2i( minimum, size );
}

bool Rect2f::contains( float x, float y )
{
	return
	(
		( x >= m_origin.x ) &&
		( x < ( m_origin.x + m_size.x ) ) &&
		( y >= m_origin.y ) &&
		( y < ( m_origin.y + m_size.y ) )
	);
}

bool Rect2f::contains( const Vector2f& point )
{
	return contains( point.x, point.y );
}

bool Rect2f::intersectRay( const Vector2f& origin, const Vector2f& direction,
	float& tNear, float& tFar, int& axis ) const
{
	// compute t to each face
	Vector2f rcpDir = 1.0f / direction;

	// intersect left and bottom
	Vector2f tBottomLeft = rcpDir * ( bottomLeft() - origin );
	// intersect right and top
	Vector2f tTopRight = rcpDir * ( topRight() - origin );

	// find the smallest and largest distances along each axis
	Vector2f tMin = MathUtils::minimum( tBottomLeft, tTopRight );
	Vector2f tMax = MathUtils::maximum( tBottomLeft, tTopRight );

	// tNear is the largest tMin
	tNear = MathUtils::maximum( tMin );

	// tFar is the smallest tMax
	tFar = MathUtils::minimum( tMax );

	bool intersected = ( tFar > tNear );
	if( intersected )
	{
		if( tNear == tMin.x || tNear == tMax.x )
		{
			axis = 0;
		}
		else
		{
			axis = 1;
		}
	}

	return intersected;
}

// static
Rect2f Rect2f::united( const Rect2f& r0, const Rect2f& r1 )
{
	Vector2f r0Min = r0.bottomLeft();
	Vector2f r0Max = r0.topRight();
	Vector2f r1Min = r1.bottomLeft();
	Vector2f r1Max = r1.topRight();

	Vector2f unitedMin( std::min( r0Min.x, r1Min.x ), std::min( r0Min.y, r1Min.y ) );
	Vector2f unitedMax( std::max( r0Max.x, r1Max.x ), std::max( r0Max.y, r1Max.y ) );

	return Rect2f( unitedMin, unitedMax - unitedMin );
}

// static
bool Rect2f::intersect( const Rect2f& r0, const Rect2f& r1 )
{
	Rect2f isect;
	return intersect( r0, r1, isect );
}

// static
bool Rect2f::intersect( const Rect2f& r0, const Rect2f& r1, Rect2f& intersection )
{
	Vector2f minimum = MathUtils::maximum( r0.bottomLeft(), r1.bottomLeft() );
	Vector2f maximum = MathUtils::minimum( r0.topRight(), r1.topRight() );

	if( minimum.x < maximum.x &&
		minimum.y < maximum.y )
	{
		intersection.m_origin = minimum;
		intersection.m_size = maximum - minimum;
		return true;
	}
	return false;
}
