#include <cmath>
#include <cstdio>

#include <math/MathUtils.h>

#include "vecmath/Quat4f.h"
#include "vecmath/Vector3f.h"
#include "vecmath/Vector4f.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
const Quat4f Quat4f::ZERO = Quat4f( 0, 0, 0, 0 );

// static
const Quat4f Quat4f::IDENTITY = Quat4f( 1, 0, 0, 0 );

Quat4f::Quat4f() :

	w( 0 ),
	x( 0 ),
	y( 0 ),
	z( 0 )

{
	w = 0;
	x = 0;
	y = 0;
	z = 0;
}

Quat4f::Quat4f( float w, float x, float y, float z ) :
	
	w( w ),
	x( x ),
	y( y ),
	z( z )
{

}

Quat4f::Quat4f( const Quat4f& rq ) :

	w( rq.w ),
	x( rq.x ),
	y( rq.y ),
	z( rq.z )

{

}

Quat4f& Quat4f::operator = ( const Quat4f& rq )
{
	if( this != ( &rq ) )
	{
		w = rq.w;
		x = rq.x;
		y = rq.y;
		z = rq.z;
	}
    return( *this );
}

Quat4f::Quat4f( const Vector3f& v ) :

	w( 0 ),
	x( v.x ),
	y( v.y ),
	z( v.z )

{

}

Quat4f::Quat4f( const Vector4f& v ) :

	w( v.x ),
	x( v.y ),
	y( v.z ),
	z( v.w )

{

}

const float& Quat4f::operator [] ( int i ) const
{
	return m_elements[ i ];
}

float& Quat4f::operator [] ( int i )
{
	return m_elements[ i ];
}

Vector3f Quat4f::xyz() const
{
	return Vector3f
	(
		x,
		y,
		z
	);
}

Vector4f Quat4f::wxyz() const
{
	return Vector4f
	(
		w,
		x,
		y,
		z
	);
}

float Quat4f::norm() const
{
	return sqrt( normSquared() );	
}

float Quat4f::normSquared() const
{
	return
	(
		w * w +
		x * x +
		y * y +
		z * z
	);
}

void Quat4f::normalize()
{
	float reciprocalNorm = 1.f / norm();

	w *= reciprocalNorm;
	x *= reciprocalNorm;
	y *= reciprocalNorm;
	z *= reciprocalNorm;
}

Quat4f Quat4f::normalized() const
{
	Quat4f q( *this );
	q.normalize();
	return q;
}

void Quat4f::conjugate()
{
	x = -x;
	y = -y;
	z = -z;
}

Quat4f Quat4f::conjugated() const
{
	return Quat4f
	(
		 w,
		-x,
		-y,
		-z
	);
}

void Quat4f::invert()
{
	Quat4f inverse = conjugated() * ( 1.0f / normSquared() );

	w = inverse.w;
	x = inverse.x;
	y = inverse.y;
	z = inverse.z;
}

Quat4f Quat4f::inverse() const
{
	return conjugated() * ( 1.0f / normSquared() );
}


Quat4f Quat4f::log() const
{
	float len =
		sqrt
		(
			x * x +
			y * y +
			z * z
		);

	if( len < 1e-6 )
	{
		return Quat4f( 0, x, y, z );
	}
	else
	{
		float coeff = acos( w ) / len;
		return Quat4f( 0, x * coeff, y * coeff, z * coeff );
	}
}

Quat4f Quat4f::exp() const
{
	float theta =
		sqrt
		(
			x * x +
			y * y +
			z * z
		);

	if( theta < 1e-6 )
	{
		return Quat4f( cos( theta ), x, y, z );
	}
	else
	{
		float coeff = sin( theta ) / theta;
		return Quat4f( cos( theta ), x * coeff, y * coeff, z * coeff );		
	}
}

Vector3f Quat4f::getAxisAngle( float* radiansOut ) const
{
	Vector4f axisAngle = getAxisAngle();
	*radiansOut = axisAngle.w;
	return axisAngle.xyz;
}

Vector4f Quat4f::getAxisAngle() const
{
	float theta = acos( w ) * 2;

	float vectorNormSquared = x * x + y * y + z * z;
	if( vectorNormSquared > 0 )
	{
		float vectorNorm = sqrt( vectorNormSquared );
		float reciprocalVectorNorm = 1.f / vectorNorm;

		return Vector4f
		(
			x * reciprocalVectorNorm,
			y * reciprocalVectorNorm,
			z * reciprocalVectorNorm,
			theta
		);
	}
	else
	{
		return Vector4f( 0, 0, 0, 0 );
	}
}

void Quat4f::setAxisAngle( float radians, const Vector3f& axis )
{
	w = cos( radians / 2 );

	float sinHalfTheta = sin( radians / 2 );
	float vectorNorm = axis.norm();
	float reciprocalVectorNorm = 1.f / vectorNorm;

	x = axis.x * sinHalfTheta * reciprocalVectorNorm;
	y = axis.y * sinHalfTheta * reciprocalVectorNorm;
	z = axis.z * sinHalfTheta * reciprocalVectorNorm;
}

void Quat4f::setAxisAngle( const Vector4f& axisAngle )
{
	setAxisAngle( axisAngle.w, axisAngle.xyz );
}

void Quat4f::print()
{
	printf( "< %.2f + %.2f i + %.2f j + %.2f k >\n",
		w, x, y, z );
}

// static
float Quat4f::dot( const Quat4f& q0, const Quat4f& q1 )
{
	return
	(
		q0.w * q1.w +
		q0.x * q1.x +
		q0.y * q1.y +
		q0.z * q1.z
	);
}

// static
Quat4f Quat4f::lerp( const Quat4f& q0, const Quat4f& q1, float alpha )
{
	return( ( q0 + alpha * ( q1 - q0 ) ).normalized() );
}

// static
Quat4f Quat4f::slerp( const Quat4f& a, const Quat4f& b, float t, bool allowFlip )
{
	float cosAngle = Quat4f::dot( a, b );

	float c1;
	float c2;

	// Linear interpolation for close orientations
	if( ( 1.0f - std::abs( cosAngle ) ) < 0.01f )
	{
		c1 = 1.0f - t;
		c2 = t;
	}
	else
	{
		// Spherical interpolation
		float angle = acos( fabs( cosAngle ) );
		float sinAngle = sin( angle );
		c1 = sin( angle * ( 1.0f - t ) ) / sinAngle;
		c2 = sin( angle * t ) / sinAngle;
	}

	// Use the shortest path
	if( allowFlip && ( cosAngle < 0.0f ) )
	{
		c1 = -c1;
	}

	return Quat4f
	(
		c1 * a.w + c2 * b.w,
		c1 * a.x + c2 * b.x,
		c1 * a.y + c2 * b.y,
		c1 * a.z + c2 * b.z
	);
}

// static
Quat4f Quat4f::squad( const Quat4f& a, const Quat4f& tanA, const Quat4f& tanB, const Quat4f& b, float t )
{
	Quat4f ab = Quat4f::slerp( a, b, t );
	Quat4f tangent = Quat4f::slerp( tanA, tanB, t, false );
	return Quat4f::slerp( ab, tangent, 2.0f * t * ( 1.0f - t ), false );
}

// static
Quat4f Quat4f::cubicInterpolate( const Quat4f& q0, const Quat4f& q1, const Quat4f& q2, const Quat4f& q3, float t )
{
	// geometric construction:
	//            t
	//   (t+1)/2     t/2
	// t+1        t	        t-1

	// bottom level
	Quat4f q0q1 = Quat4f::slerp( q0, q1, t + 1 );
	Quat4f q1q2 = Quat4f::slerp( q1, q2, t );
	Quat4f q2q3 = Quat4f::slerp( q2, q3, t - 1 );

	// middle level
	Quat4f q0q1_q1q2 = Quat4f::slerp( q0q1, q1q2, 0.5f * ( t + 1 ) );
	Quat4f q1q2_q2q3 = Quat4f::slerp( q1q2, q2q3, 0.5f * t );

	// top level
	return Quat4f::slerp( q0q1_q1q2, q1q2_q2q3, t );
}

// static
Quat4f Quat4f::logDifference( const Quat4f& a, const Quat4f& b )
{
	Quat4f diff = a.inverse() * b;
	diff.normalize();
	return diff.log();
}

// static
Quat4f Quat4f::squadTangent( const Quat4f& before, const Quat4f& center, const Quat4f& after )
{
	Quat4f l1 = Quat4f::logDifference( center, before );
	Quat4f l2 = Quat4f::logDifference( center, after );
	
	Quat4f e;
	for( int i = 0; i < 4; ++i )
	{
		e[ i ] = -0.25f * ( l1[ i ] + l2[ i ] );
	}
	e = center * ( e.exp() );

	return e;
}

// static
Quat4f Quat4f::fromRotationMatrix( const Matrix3f& m )
{
	float x;
	float y;
	float z;
	float w;

	// Compute one plus the trace of the matrix
	float onePlusTrace = 1.0f + m( 0, 0 ) + m( 1, 1 ) + m( 2, 2 );

	if( onePlusTrace > 1e-5 )
	{
		// Direct computation
		float s = sqrt( onePlusTrace ) * 2.0f;
		x = ( m( 2, 1 ) - m( 1, 2 ) ) / s;
		y = ( m( 0, 2 ) - m( 2, 0 ) ) / s;
		z = ( m( 1, 0 ) - m( 0, 1 ) ) / s;
		w = 0.25f * s;
	}
	else
	{
		// Computation depends on major diagonal term
		if( ( m( 0, 0 ) > m( 1, 1 ) ) & ( m( 0, 0 ) > m( 2, 2 ) ) )
		{
			float s = sqrt( 1.0f + m( 0, 0 ) - m( 1, 1 ) - m( 2, 2 ) ) * 2.0f;
			x = 0.25f * s;
			y = ( m( 0, 1 ) + m( 1, 0 ) ) / s;
			z = ( m( 0, 2 ) + m( 2, 0 ) ) / s;
			w = ( m( 1, 2 ) - m( 2, 1 ) ) / s;
		}
		else if( m( 1, 1 ) > m( 2, 2 ) )
		{
			float s = sqrt( 1.0f + m( 1, 1 ) - m( 0, 0 ) - m( 2, 2 ) ) * 2.0f;
			x = ( m( 0, 1 ) + m( 1, 0 ) ) / s;
			y = 0.25f * s;
			z = ( m( 1, 2 ) + m( 2, 1 ) ) / s;
			w = ( m( 0, 2 ) - m( 2, 0 ) ) / s;
		}
		else
		{
			float s = sqrt( 1.0f + m( 2, 2 ) - m( 0, 0 ) - m( 1, 1 ) ) * 2.0f;
			x = ( m( 0, 2 ) + m( 2, 0 ) ) / s;
			y = ( m( 1, 2 ) + m( 2, 1 ) ) / s;
			z = 0.25f * s;
			w = ( m( 0, 1 ) - m( 1, 0 ) ) / s;
		}
	}

	Quat4f q( w, x, y, z );
	return q.normalized();
}

// static
Quat4f Quat4f::fromRotatedBasis( const Vector3f& x, const Vector3f& y, const Vector3f& z )
{
	return fromRotationMatrix( Matrix3f( x, y, z ) );
}

// static
Quat4f Quat4f::randomRotation( float u0, float u1, float u2 )
{
	float z = u0;
	float theta = static_cast< float >( 2.f * MathUtils::PI * u1 );
	float r = sqrt( 1.f - z * z );
	float w = static_cast< float >( MathUtils::PI * u2 );

	return Quat4f
	(
		cos( w ),
		sin( w ) * cos( theta ) * r,
		sin( w ) * sin( theta ) * r,
		sin( w ) * z
	);
}

Vector3f Quat4f::rotateVector( const Vector3f& v )
{	
	return ( ( *this ) * Quat4f( v ) * conjugated() ).xyz();
}

//////////////////////////////////////////////////////////////////////////
// Operators
//////////////////////////////////////////////////////////////////////////

Quat4f operator + ( const Quat4f& q0, const Quat4f& q1 )
{
	return Quat4f
	(
		q0.w + q1.w,
		q0.x + q1.x,
		q0.y + q1.y,
		q0.z + q1.z
	);
}

Quat4f operator - ( const Quat4f& q0, const Quat4f& q1 )
{
	return Quat4f
	(
		q0.w - q1.w,
		q0.x - q1.x,
		q0.y - q1.y,
		q0.z - q1.z
	);
}

Quat4f operator * ( const Quat4f& q0, const Quat4f& q1 )
{
	return Quat4f
	(
		q0.w * q1.w - q0.x * q1.x - q0.y * q1.y - q0.z * q1.z,
		q0.w * q1.x + q0.x * q1.w + q0.y * q1.z - q0.z * q1.y,
		q0.w * q1.y - q0.x * q1.z + q0.y * q1.w + q0.z * q1.x,
		q0.w * q1.z + q0.x * q1.y - q0.y * q1.x + q0.z * q1.w
	);
}

Quat4f operator * ( float f, const Quat4f& q )
{
	return Quat4f
	(
		f * q.w,
		f * q.x,
		f * q.y,
		f * q.z
	);
}

Quat4f operator * ( const Quat4f& q, float f )
{
	return Quat4f
	(
		f * q.w,
		f * q.x,
		f * q.y,
		f * q.z
	);
}
