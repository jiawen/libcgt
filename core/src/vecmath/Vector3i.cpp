#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <QString>

#include "vecmath/Vector3i.h"

#include "vecmath/Vector2i.h"
#include "vecmath/Vector3f.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

Vector3i::Vector3i()
{
    m_elements[ 0 ] = 0;
    m_elements[ 1 ] = 0;
    m_elements[ 2 ] = 0;
}

Vector3i::Vector3i( int i )
{
    m_elements[ 0 ] = i;
    m_elements[ 1 ] = i;
    m_elements[ 2 ] = i;
}

Vector3i::Vector3i( std::initializer_list< int > xyz )
{
    m_elements[ 0 ] = *( xyz.begin( ) );
    m_elements[ 1 ] = *( xyz.begin( ) + 1 );
    m_elements[ 2 ] = *( xyz.begin( ) + 2 );
}

Vector3i::Vector3i( const Vector2i& xy, int z )
{
	m_elements[0] = xy.x;
	m_elements[1] = xy.y;
	m_elements[2] = z;
}

Vector3i::Vector3i( int x, const Vector2i& yz )
{
	m_elements[0] = x;
	m_elements[1] = yz.x;
	m_elements[2] = yz.y;
}

Vector3i::Vector3i( const Vector3i& rv )
{
	m_elements[0] = rv.m_elements[0];
	m_elements[1] = rv.m_elements[1];
	m_elements[2] = rv.m_elements[2];
}

Vector3i& Vector3i::operator = ( const Vector3i& rv )
{
	if( this != &rv )
	{
		m_elements[0] = rv.m_elements[0];
		m_elements[1] = rv.m_elements[1];
		m_elements[2] = rv.m_elements[2];
	}
	return *this;
}

const int& Vector3i::operator [] ( int i ) const
{
	return m_elements[ i ];
}

int& Vector3i::operator [] ( int i )
{
	return m_elements[ i ];
}

Vector2i Vector3i::xy() const
{
    return{ x, y };
}

Vector2i Vector3i::yz() const
{
    return{ y, z };
}

Vector2i Vector3i::zx() const
{
    return{ z, x };
}

Vector2i Vector3i::yx() const
{
    return{ y, x };
}

Vector2i Vector3i::zy() const
{
    return{ z, y };
}

Vector2i Vector3i::xz() const
{
    return{ x, z };
}

Vector3i Vector3i::xyz() const
{
    return { m_elements[ 0 ], m_elements[ 1 ], m_elements[ 2 ] };
}

Vector3i Vector3i::yzx() const
{
    return { m_elements[ 1 ], m_elements[ 2 ], m_elements[ 0 ] };
}

Vector3i Vector3i::zxy() const
{
    return { m_elements[ 2 ], m_elements[ 0 ], m_elements[ 1 ] };
}

float Vector3i::norm() const
{
	return sqrt( static_cast< float >( normSquared() ) );
}

int Vector3i::normSquared() const
{
	return( x * x + y * y + z * z );
}

Vector3f Vector3i::normalized() const
{
	float rLength = 1.f / norm();

	return Vector3f
	(
		rLength * m_elements[ 0 ],
		rLength * m_elements[ 1 ],
		rLength * m_elements[ 2 ]
	);
}

void Vector3i::negate()
{
	m_elements[0] = -m_elements[0];
	m_elements[1] = -m_elements[1];
	m_elements[2] = -m_elements[2];
}

Vector3i::operator const int* () const
{
	return m_elements;
}

Vector3i::operator int* ()
{
	return m_elements;
}

QString Vector3i::toString() const
{
	QString out;

	out.append( "( " );
	out.append( QString( "%1" ).arg( x, 10 ) );
	out.append( QString( "%1" ).arg( y, 10 ) );
	out.append( QString( "%1" ).arg( z, 10 ) );
	out.append( " )" );

	return out;
}

// static
int Vector3i::dot( const Vector3i& v0, const Vector3i& v1 )
{
	return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

// static
Vector3i Vector3i::cross( const Vector3i& v0, const Vector3i& v1 )
{
	return
    {
        v0.y * v1.z - v0.z * v1.y,
        v0.z * v1.x - v0.x * v1.z,
        v0.x * v1.y - v0.y * v1.x
    };
}

// static
Vector3f Vector3i::lerp( const Vector3i& v0, const Vector3i& v1, float alpha )
{
	return alpha * ( v1 - v0 ) + Vector3f( v0 );
}

//////////////////////////////////////////////////////////////////////////
// Operators
//////////////////////////////////////////////////////////////////////////

bool operator == ( const Vector3i& v0, const Vector3i& v1 )
{
	return
	(
		v0.x == v1.x &&
		v0.y == v1.y &&
		v0.z == v1.z
	);
}

bool operator != ( const Vector3i& v0, const Vector3i& v1 )
{
	return !( v0 == v1 );
}

Vector3i operator + ( const Vector3i& v0, const Vector3i& v1 )
{
	return { v0.x + v1.x, v0.y + v1.y, v0.z + v1.z };
}

Vector3i operator - ( const Vector3i& v0, const Vector3i& v1 )
{
	return { v0.x - v1.x, v0.y - v1.y, v0.z - v1.z };
}

Vector3i operator * ( const Vector3i& v0, const Vector3i& v1 )
{
	return { v0.x * v1.x, v0.y * v1.y, v0.z * v1.z };
}

Vector3i operator - ( const Vector3i& v )
{
	return { -v.x, -v.y, -v.z };
}

Vector3i operator * ( int c, const Vector3i& v )
{
	return { c * v.x, c * v.y, c * v.z };
}

Vector3i operator * ( const Vector3i& v, int c )
{
	return { c * v.x, c * v.y, c * v.z };
}

Vector3f operator * ( float f, const Vector3i& v )
{
    return Vector3f( f * v.x, f * v.y, f * v.z );
}

Vector3f operator * ( const Vector3i& v, float f )
{
    return Vector3f( f * v.x, f * v.y, f * v.z );
}

Vector3i operator / ( const Vector3i& v, int c )
{
	return { v.x / c, v.y / c, v.z / c };
}

Vector3i operator / ( const Vector3i& v0, const Vector3i& v1 )
{
	return { v0.x / v1.x, v0.y / v1.y, v0.z / v1.z };
}