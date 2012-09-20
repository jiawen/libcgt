#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <QString>

#include "vecmath/Vector3f.h"
#include "vecmath/Vector3d.h"
#include "vecmath/Vector3i.h"
#include "vecmath/Vector2f.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
const Vector3f Vector3f::ZERO = Vector3f( 0, 0, 0 );

// static
const Vector3f Vector3f::UP = Vector3f( 0, 1, 0 );

// static
const Vector3f Vector3f::RIGHT = Vector3f( 1, 0, 0 );

// static
const Vector3f Vector3f::FORWARD = Vector3f( 0, 0, -1 );

Vector3f::Vector3f( const Vector2f& xy, float z )
{
	m_elements[0] = xy.x;
	m_elements[1] = xy.y;
	m_elements[2] = z;
}

Vector3f::Vector3f( float x, const Vector2f& yz )
{
	m_elements[0] = x;
	m_elements[1] = yz.x;
	m_elements[2] = yz.y;
}

Vector3f::Vector3f( const Vector3d& rv )
{
	m_elements[0] = static_cast< float >( rv.x );
	m_elements[1] = static_cast< float >( rv.y );
	m_elements[2] = static_cast< float >( rv.z );
}

Vector3f::Vector3f( const Vector3i& rv )
{
	m_elements[0] = static_cast< float >( rv.x );
	m_elements[1] = static_cast< float >( rv.y );
	m_elements[2] = static_cast< float >( rv.z );
}

Vector3f& Vector3f::operator = ( const Vector3d& rv )
{
	m_elements[ 0 ] = static_cast< float >( rv.x );
	m_elements[ 1 ] = static_cast< float >( rv.y );
	m_elements[ 2 ] = static_cast< float >( rv.z );

	return *this;
}

Vector3f& Vector3f::operator = ( const Vector3i& rv )
{
	m_elements[ 0 ] = static_cast< float >( rv.x );
	m_elements[ 1 ] = static_cast< float >( rv.y );
	m_elements[ 2 ] = static_cast< float >( rv.z );

	return *this;
}

Vector2f Vector3f::xy() const
{
	return Vector2f( x, y );
}

Vector2f Vector3f::xz() const
{
	return Vector2f( x, z );
}

Vector2f Vector3f::yz() const
{
	return Vector2f( y, z );
}

Vector3f Vector3f::xyz() const
{
	return Vector3f( x, y, z );
}

Vector3f Vector3f::yzx() const
{
	return Vector3f( y, z, x );
}

Vector3f Vector3f::zxy() const
{
	return Vector3f( z, x, y );
}

float Vector3f::norm() const
{
	return sqrt( normSquared() );
}

float Vector3f::normSquared() const
{
	return( x * x + y * y + z * z );
}

void Vector3f::normalize()
{
	float rcpNorm = 1.0f / norm();
	x *= rcpNorm;
	y *= rcpNorm;
	z *= rcpNorm;
}

Vector3f Vector3f::normalized() const
{
	float rcpNorm = 1.0f / norm();
	return Vector3f
	(
		x * rcpNorm,
		y * rcpNorm,
		z * rcpNorm
	);
}

void Vector3f::homogenize()
{
	if( z != 0 )
	{
		float rcpZ = 1.0f / z;
		x *= rcpZ;
		y *= rcpZ;
		z = 1;
	}
}

Vector3f Vector3f::homogenized() const
{
	if( z != 0 )
	{
		float rcpZ = 1.0f / z;
		return Vector3f( rcpZ * x, rcpZ * y, 1 );
	}
	else
	{
		return Vector3f( x, y, z );
	}
}

void Vector3f::negate()
{
	x = -x;
	y = -y;
	z = -z;
}

QString Vector3f::toString() const
{
	QString out;

	out.append( "( " );
	out.append( QString( "%1, " ).arg( x, 10, 'g', 4 ) );
	out.append( QString( "%1, " ).arg( y, 10, 'g', 4 ) );
	out.append( QString( "%1" ).arg( z, 10, 'g', 4 ) );
	out.append( " )" );

	return out;
}
