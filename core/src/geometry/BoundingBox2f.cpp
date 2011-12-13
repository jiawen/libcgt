#include "geometry/BoundingBox2f.h"

#include <algorithm>
#include <cfloat>
#include <cstdio>

BoundingBox2f::BoundingBox2f() :

	m_min( FLT_MAX, FLT_MAX ),
	m_max( FLT_MIN, FLT_MIN )

{

}

BoundingBox2f::BoundingBox2f( float minX, float minY, float maxX, float maxY ) :

	m_min( minX, minY ),
	m_max( maxX, maxY )

{

}

BoundingBox2f::BoundingBox2f( const Vector2f& min, const Vector2f& max ) :

	m_min( min ),
	m_max( max )

{

}

BoundingBox2f::BoundingBox2f( const BoundingBox2f& rb ) :

	m_min( rb.m_min ),
	m_max( rb.m_max )
	
{

}

BoundingBox2f& BoundingBox2f::operator = ( const BoundingBox2f& rb )
{
	if( this != &rb )
	{
		m_min = rb.m_min;
		m_max = rb.m_max;
	}
	return *this;
}

void BoundingBox2f::print()
{
	printf( "min: " );
	m_min.print();
	printf( "max: " );
	m_max.print();
}

Vector2f& BoundingBox2f::minimum()
{
	return m_min;
}

Vector2f& BoundingBox2f::maximum()
{
	return m_max;
}

Vector2f BoundingBox2f::minimum() const
{
	return m_min;
}

Vector2f BoundingBox2f::maximum() const
{
	return m_max;
}

Vector2f BoundingBox2f::range() const
{
	return( m_max - m_min );
}

Vector2f BoundingBox2f::center() const
{
	return( 0.5f * ( m_max + m_min ) );
}

// static
BoundingBox2f BoundingBox2f::merge( const BoundingBox2f& b0, const BoundingBox2f& b1 )
{
	Vector2f b0Min = b0.minimum();
	Vector2f b0Max = b0.maximum();
	Vector2f b1Min = b1.minimum();
	Vector2f b1Max = b1.maximum();

	Vector2f mergedMin( std::min( b0Min.x, b1Min.x ), std::min( b0Min.y, b1Min.y ) );
	Vector2f mergedMax( std::max( b0Max.x, b1Max.x ), std::max( b0Max.y, b1Max.y ) );

	return BoundingBox2f( mergedMin, mergedMax );
}
