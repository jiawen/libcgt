#include "geometry/BoundingBox3f.h"

#include <algorithm>
#include <cfloat>
#include <cstdio>

using namespace std;

BoundingBox3f::BoundingBox3f() :
	
	m_min( FLT_MAX, FLT_MAX, FLT_MAX ),
	m_max( FLT_MIN, FLT_MIN, FLT_MIN )

{

}

BoundingBox3f::BoundingBox3f( float minX, float minY, float minZ,
							 float maxX, float maxY, float maxZ ) :

	m_min( minX, minY, minZ ),
	m_max( maxX, maxY, maxZ )

{

}

BoundingBox3f::BoundingBox3f( const Vector3f& min, const Vector3f& max ) :

	m_min( min ),
	m_max( max )

{

}

BoundingBox3f::BoundingBox3f( const BoundingBox3f& rb ) :

	m_min( rb.m_min ),
	m_max( rb.m_max )

{

}

BoundingBox3f& BoundingBox3f::operator = ( const BoundingBox3f& rb )
{
	if( this != &rb )
	{
		m_min = rb.m_min;
		m_max = rb.m_max;
	}
	return *this;
}

void BoundingBox3f::print()
{
	printf( "min: " );
	m_min.print();
	printf( "max: " );
	m_max.print();
}

Vector3f& BoundingBox3f::minimum()
{
	return m_min;
}

Vector3f& BoundingBox3f::maximum()
{
	return m_max;
}

Vector3f BoundingBox3f::minimum() const
{
	return m_min;
}

Vector3f BoundingBox3f::maximum() const
{
	return m_max;
}

Vector3f BoundingBox3f::range() const
{
	return( m_max - m_min );
}

Vector3f BoundingBox3f::center() const
{
	return( 0.5 * ( m_max + m_min ) );
}

float BoundingBox3f::radius() const
{
	Vector3f diameter = range();
	return min( diameter.x(), min( diameter.y(), diameter.z() ) );
}

QVector< Vector3f > BoundingBox3f::corners() const
{
	QVector< Vector3f > out( 8 );

	for( int i = 0; i < 8; ++i )
	{
		out[ i ] =
			Vector3f
			(
				( i & 1 ) ? minimum().x() : maximum().x(),
				( i & 2 ) ? minimum().y() : maximum().y(),
				( i & 4 ) ? minimum().z() : maximum().z()
			);
	}

	return out;
}

bool BoundingBox3f::overlaps( const BoundingBox3f& other )
{
	bool bOverlapsInDirection[3];

	Vector3f otherMin = other.minimum();
	Vector3f otherMax = other.maximum();

	for( int i = 0; i < 3; ++i )
	{
		bool bMinInside0 = ( otherMin[i] >= m_min[i] ) && ( otherMin[i] <= m_max[i] );
		bool bMinInside1 = ( m_min[i] >= otherMin[i] ) && ( m_min[i] <= otherMax[i] );

		bool bMaxInside0 = ( otherMax[i] >= m_min[i] ) && ( otherMax[i] <= m_max[i] );
		bool bMaxInside1 = ( m_max[i] >= otherMin[i] ) && ( m_max[i] <= otherMax[i] );

		bool bMinInside = bMinInside0 || bMinInside1;
		bool bMaxInside = bMaxInside0 || bMaxInside1;

		bOverlapsInDirection[i] = bMinInside || bMaxInside;
	}

	return bOverlapsInDirection[0] && bOverlapsInDirection[1] && bOverlapsInDirection[2];
}

// static
BoundingBox3f BoundingBox3f::unite( const BoundingBox3f& b0, const BoundingBox3f& b1 )
{
    Vector3f b0Min = b0.minimum();
    Vector3f b0Max = b0.maximum();
    Vector3f b1Min = b1.minimum();
    Vector3f b1Max = b1.maximum();

    Vector3f newMin( min( b0Min.x(), b1Min.x() ), min( b0Min.y(), b1Min.y() ), min( b0Min.z(), b1Min.z() ) );
    Vector3f newMax( max( b0Max.x(), b1Max.x() ), max( b0Max.y(), b1Max.y() ), max( b0Max.z(), b1Max.z() ) );

    return BoundingBox3f( newMin, newMax );
}

// static
BoundingBox3f BoundingBox3f::intersect( const BoundingBox3f& b0, const BoundingBox3f& b1 )
{
    Vector3f b0Min = b0.minimum();
    Vector3f b0Max = b0.maximum();
    Vector3f b1Min = b1.minimum();
    Vector3f b1Max = b1.maximum();

    Vector3f newMin( max( b0Min.x(), b1Min.x() ), max( b0Min.y(), b1Min.y() ), max( b0Min.z(), b1Min.z() ) );
    Vector3f newMax( min( b0Max.x(), b1Max.x() ), min( b0Max.y(), b1Max.y() ), min( b0Max.z(), b1Max.z() ) );

    for(int i = 0; i < 3; ++i)
        newMax[i] = max(newMax[i], newMin[i]);

    return BoundingBox3f( newMin, newMax );
}

void BoundingBox3f::enlarge( const Vector3f& p )
{
	m_min.x() = min( p.x(), m_min.x() );
	m_min.y() = min( p.y(), m_min.y() );
	m_min.z() = min( p.z(), m_min.z() );
	m_max.x() = max( p.x(), m_max.x() );
	m_max.y() = max( p.y(), m_max.y() );
	m_max.z() = max( p.z(), m_max.z() );
}