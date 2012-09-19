#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <QString>

#include "vecmath/Vector4f.h"
#include "vecmath/Vector2f.h"
#include "vecmath/Vector3f.h"
#include "vecmath/Vector4d.h"
#include "vecmath/Vector4i.h"

Vector4f::Vector4f( const Vector2f& _xy, float _z, float _w ) :

	x( _xy.x ),
	y( _xy.y ),
	z( _z ),
	w( _w )

{

}

Vector4f::Vector4f( float _x, const Vector2f& _yz, float _w ) :

	x( _x ),
	y( _yz.x ),
	z( _yz.y ),
	w( _w )

{

}

Vector4f::Vector4f( float _x, float _y, const Vector2f& _zw ) :

	x( _x ),
	y( _y ),
	z( _zw.x ),
	w( _zw.y )

{

}

Vector4f::Vector4f( const Vector2f& _xy, const Vector2f& _zw ) :

	x( _xy.x ),
	y( _xy.y ),
	z( _zw.x ),
	w( _zw.y )

{

}

Vector4f::Vector4f( const Vector3f& _xyz, float _w ) :

	x( _xyz.x ),
	y( _xyz.y ),
	z( _xyz.z ),
	w( _w )

{

}

Vector4f::Vector4f( float _x, const Vector3f& _yzw ) :

	x( _x ),
	y( _yzw.x ),
	z( _yzw.y ),
	w( _yzw.z )

{

}

Vector4f::Vector4f( const Vector4d& v ) :

	x( static_cast< float >( v.x ) ),
	y( static_cast< float >( v.y ) ),
	z( static_cast< float >( v.z ) ),
	w( static_cast< float >( v.w ) )

{

}

Vector4f::Vector4f( const Vector4i& v ) :

	x( static_cast< float >( v.x ) ),
	y( static_cast< float >( v.y ) ),
	z( static_cast< float >( v.z ) ),
	w( static_cast< float >( v.w ) )

{

}

Vector2f Vector4f::xy() const
{
	return Vector2f( x, y );
}

Vector2f Vector4f::yz() const
{
	return Vector2f( y, z );
}

Vector2f Vector4f::zw() const
{
	return Vector2f( z, w );
}

Vector2f Vector4f::wx() const
{
	return Vector2f( w, x );
}

Vector3f Vector4f::xyz() const
{
	return Vector3f( x, y, z );
}

Vector3f Vector4f::yzw() const
{
	return Vector3f( y, z, w );
}

Vector3f Vector4f::zwx() const
{
	return Vector3f( z, w, x );
}

Vector3f Vector4f::wxy() const
{
	return Vector3f( w, x, y );
}

Vector3f Vector4f::xyw() const
{
	return Vector3f( x, y, w );
}

Vector3f Vector4f::yzx() const
{
	return Vector3f( y, z, x );
}

Vector3f Vector4f::zwy() const
{
	return Vector3f( z, w, y );
}

Vector3f Vector4f::wxz() const
{
	return Vector3f( w, x, z );
}

float Vector4f::norm() const
{
	return sqrt( normSquared() );
}

float Vector4f::normSquared() const
{
	return x * x + y * y + z * z + w * w;
}

void Vector4f::normalize()
{
	float rcpNorm = 1.0f / norm();
	x *= rcpNorm;
	y *= rcpNorm;
	z *= rcpNorm;
	w *= rcpNorm;
}

Vector4f Vector4f::normalized() const
{
	float rcpNorm = 1.0f / norm();
	return Vector4f
	(
		x * rcpNorm, 
		y * rcpNorm, 
		z * rcpNorm, 
		w * rcpNorm
	);
}

void Vector4f::homogenize()
{
	if( w != 0 )
	{
		float rcpW = 1.0f / w;
		x *= rcpW;
		y *= rcpW;
		z *= rcpW;
		w = 1;
	}
}

Vector4f Vector4f::homogenized() const
{
	if( w != 0 )
	{
		float rcpW = 1.0f / w;
		return Vector4f( rcpW * x, rcpW * y, rcpW * z, 1 );
	}
	else
	{
		return Vector4f( x, y, z, w );
	}
}

void Vector4f::negate()
{
	x = -x;
	y = -y;
	z = -z;
	w = -w;
}

float Vector4f::minimum() const
{
	return std::min( x, std::min( y, std::min( z, w ) ) );
}

float Vector4f::maximum() const
{
	return std::max( x, std::max( y, std::max( z, w ) ) );
}

// static
Vector4f Vector4f::minimum( const Vector4f& v0, const Vector4f& v1 )
{
	return Vector4f
	(
		std::min( v0.x, v1.x ),
		std::min( v0.y, v1.y ),
		std::min( v0.z, v1.z ),
		std::min( v0.w, v1.w )
	);
}

// static
Vector4f Vector4f::maximum( const Vector4f& v0, const Vector4f& v1 )
{
	return Vector4f
	(
		std::max( v0.x, v1.x ),
		std::max( v0.y, v1.y ),
		std::max( v0.z, v1.z ),
		std::max( v0.w, v1.w )
	);
}

QString Vector4f::toString() const
{
	QString out;

	out.append( "( " );
	out.append( QString( "%1, " ).arg( x, 10, 'g', 4 ) );
	out.append( QString( "%1, " ).arg( y, 10, 'g', 4 ) );
	out.append( QString( "%1, " ).arg( z, 10, 'g', 4 ) );
	out.append( QString( "%1" ).arg( w, 10, 'g', 4 ) );
	out.append( " )" );

	return out;
}

// static
float Vector4f::dot( const Vector4f& v0, const Vector4f& v1 )
{
	return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z + v0.w * v1.w;
}

// static
Vector4f Vector4f::lerp( const Vector4f& v0, const Vector4f& v1, float alpha )
{
	return alpha * ( v1 - v0 ) + v0;
}
