#include <cassert>
#include <cmath>

#include "math/Arithmetic.h"

// static
int Arithmetic::mod( int x, int n )
{
	return ( ( x % n ) + n ) % n;
}

// static
Vector2i Arithmetic::mod( const Vector2i& v, const Vector2i& n )
{
    return{ mod( v.x, n.x ), mod( v.y, n.y ) };
}

// static
Vector3i Arithmetic::mod( const Vector3i& v, const Vector3i& n )
{
	return
    {
        mod( v.x, n.x ),
        mod( v.y, n.y ),
        mod( v.z, n.z )
    };
}

// static
Vector4i Arithmetic::mod( const Vector4i& v, const Vector4i& n )
{
	return
    {
        mod( v.x, n.x ),
        mod( v.y, n.y ),
        mod( v.z, n.z ),
        mod( v.w, n.w )
    };
}

// static
int Arithmetic::sign( int x )
{
	if( x < 0 )
	{
		return -1;
	}
	if( x > 0 )
	{
		return 1;
	}
	return 0;
}

// static
int Arithmetic::sign( float x )
{
	if( x < 0 )
	{
		return -1;
	}
	if( x > 0 )
	{
		return 1;
	}
	return 0;
}

// static
int Arithmetic::sign( double x )
{
	if( x < 0 )
	{
		return -1;
	}
	if( x > 0 )
	{
		return 1;
	}
	return 0;
}

// static
Vector2i Arithmetic::sign( const Vector2f& v )
{
    return{ sign( v.x ), sign( v.y ) };
}

// static
Vector3i Arithmetic::sign( const Vector3f& v )
{
	return{ sign( v.x ), sign( v.y ), sign( v.z ) };
}

// static
Vector4i Arithmetic::sign( const Vector4f& v )
{
	return{ sign( v.x ), sign( v.y ), sign( v.z ), sign( v.w ) };
}

// static
bool Arithmetic::sameSign( float x, float y )
{	
	return sign( x ) == sign( y );
}

// static
float Arithmetic::divideIntsToFloat( int numerator, int denominator )
{
	float fNumerator = static_cast< float >( numerator );
	return fNumerator / denominator;
}

// static
int Arithmetic::divideIntsToFloatAndRound( int numerator, int denominator )
{
	float f = Arithmetic::divideIntsToFloat( numerator, denominator );
	return Arithmetic::roundToInt( f );
}

// static
float Arithmetic::percentage( int numerator, int denominator )
{
	return 100.0f * divideIntsToFloat( numerator, denominator );
}

// static		
int Arithmetic::numBins( int arraySize, int binSize )
{
	return ceilToInt( divideIntsToFloat( arraySize, binSize ) );
}

// static
bool Arithmetic::isPowerOfTwo( int x )
{
	if( x <= 0 )
	{
		return 0;
	}
	else
	{
		return( ( x & ( x - 1 ) ) == 0 );
	}
}

// static
int Arithmetic::roundToInt( float x )
{
	return static_cast< int >( x + 0.5f );
}

// static
int Arithmetic::floatToInt( float x )
{
	return static_cast< int >( x );
}

// static
int Arithmetic::floorToInt( float x )
{
	return static_cast< int >( ::floor( x ) );
}

// static
int Arithmetic::ceilToInt( float x )
{
	return static_cast< int >( ::ceil( x ) );
}

// static
int Arithmetic::roundToInt( double x )
{
	// 2^52 * 1.5, uses limited precision to floor
	x = x + Arithmetic::s_dDoubleMagic;
	return( ( int* ) &x )[0];
}

// static
int Arithmetic::doubleToInt( double x )
{
	return( ( x < 0 ) ?
		Arithmetic::roundToInt( x + s_dDoubleMagicRoundEpsilon ) :
		Arithmetic::roundToInt( x - s_dDoubleMagicRoundEpsilon ) );
}

// static
int Arithmetic::floorToInt( double x )
{
	return Arithmetic::roundToInt( x - s_dDoubleMagicRoundEpsilon );
}

// static
int Arithmetic::ceilToInt( double x )
{
	return Arithmetic::roundToInt( x + s_dDoubleMagicRoundEpsilon );
}

// static
Vector2f Arithmetic::floor( const Vector2f& v )
{
	return Vector2f( ::floor( v.x ), ::floor( v.y ) );
}

// static
Vector2f Arithmetic::ceil( const Vector2f& v )
{
	return Vector2f( ::ceil( v.x ), ::ceil( v.y ) );
}

// static
Vector2i Arithmetic::roundToInt( const Vector2f& v )
{
    return{ roundToInt( v.x ), roundToInt( v.y ) };
}

// static
Vector2i Arithmetic::floorToInt( const Vector2f& v )
{
    return{ floorToInt( v.x ), floorToInt( v.y ) };
}

// static
Vector2i Arithmetic::ceilToInt( const Vector2f& v )
{
    return{ ceilToInt( v.x ), ceilToInt( v.y ) };
}

// static
Vector3f Arithmetic::floor( const Vector3f& v )
{
	return Vector3f( ::floor( v.x ), ::floor( v.y ), ::floor( v.z ) );
}

// static
Vector3f Arithmetic::ceil( const Vector3f& v )
{
	return Vector3f( ::ceil( v.x ), ::ceil( v.y ), ::ceil( v.z ) );
}

// static
Vector3i Arithmetic::roundToInt( const Vector3f& v )
{
	return{ roundToInt( v.x ), roundToInt( v.y ), roundToInt( v.z ) };
}

// static
Vector3i Arithmetic::floorToInt( const Vector3f& v )
{
	return{ floorToInt( v.x ), floorToInt( v.y ), floorToInt( v.z ) };
}

// static
Vector3i Arithmetic::ceilToInt( const Vector3f& v )
{
    return{ ceilToInt( v.x ), ceilToInt( v.y ), ceilToInt( v.z ) };
}

// static
Vector4f Arithmetic::floor( const Vector4f& v )
{
	return Vector4f( ::floor( v.x ), ::floor( v.y ), ::floor( v.z ), ::floor( v.w ) );
}

// static
Vector4f Arithmetic::ceil( const Vector4f& v )
{
	return Vector4f( ::ceil( v.x ), ::ceil( v.y ), ::ceil( v.z ), ::ceil( v.w ) );
}

// static
Vector4i Arithmetic::roundToInt( const Vector4f& v )
{
	return{ roundToInt( v.x ), roundToInt( v.y ), roundToInt( v.z ), roundToInt( v.w ) };
}

// static
Vector4i Arithmetic::floorToInt( const Vector4f& v )
{
	return{ floorToInt( v.x ), floorToInt( v.y ), floorToInt( v.z ), floorToInt( v.w ) };
}

// static
Vector4i Arithmetic::ceilToInt( const Vector4f& v )
{
	return{ ceilToInt( v.x ), ceilToInt( v.y ), ceilToInt( v.z ), ceilToInt( v.w ) };
}

// static
int Arithmetic::log2( int x )
{
    int output = 0;
    x >>= 1;
    while( x > 0 )
    {
        ++output;
        x >>= 1;
    }
    return output;
}

// static
float Arithmetic::log2( float x )
{	
	return( logf( x ) * Arithmetic::s_fReciprocalLog2 );
}

// static
int Arithmetic::log2ToInt( float x )
{
	return( ( *( int* )( &x ) ) >> 23 ) - 127;
}

// static
uint32_t Arithmetic::roundUpToNearestPowerOfTwo( uint32_t v )
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	
	return( v + 1 );
}

// static
int Arithmetic::roundUpToNearestMultipleOf4( int x )
{
	return ( x + 3 ) & ( ~( 0x3 ) );
}

// static
int Arithmetic::roundUpToNearestMultipleOf8( int x )
{
	return ( x + 7 ) & ( ~( 0x7 ) );
}

// static
int Arithmetic::roundUpToNearestMultipleOf16( int x )
{
	return ( x + 15 ) & ( ~( 0xf ) );
}

// static
int Arithmetic::roundUpToNearestMultipleOf128( int x )
{
	return ( x + 127 ) & ( ~( 0x7f ) );
}

// static
int Arithmetic::roundUpToNearestMultipleOf256( int x )
{
	return ( x + 255 ) & ( ~( 0xff ) );
}

// static
int Arithmetic::findNextPerfectSquare( int x )
{
	int y = x;
	while( !isPerfectSquare( y ) )
	{
		++y;
	}

	return y;
}

// static
int Arithmetic::findNextPerfectSquare( int x, int& sqrtOut )
{
	int y = x;
	while( !isPerfectSquare( y, sqrtOut ) )
	{
		++y;
	}

	return y;
}

// static
bool Arithmetic::isPerfectSquare( int x )
{
	int s;
	return isPerfectSquare( x, s );
}

// static
bool Arithmetic::isPerfectSquare( int x, int& sqrtOut )
{
	float fx = static_cast< float >( x );
	float sqrtFX = sqrt( fx );
	int sqrtXLower = Arithmetic::floorToInt( sqrtFX );
	int sqrtXUpper = sqrtXLower + 1;

	if( sqrtXLower * sqrtXLower == x )
	{
		sqrtOut = sqrtXLower;
		return true;
	}

	if( sqrtXUpper * sqrtXUpper == x )
	{
		sqrtOut = sqrtXUpper;
		return true;
	}

	return false;
}

// static
int Arithmetic::integerSquareRoot( int x )
{
	float fx = static_cast< float >( x );
	float sqrtFX = sqrt( fx );
	int sqrtXLower = Arithmetic::floorToInt( sqrtFX );
	int sqrtXUpper = sqrtXLower + 1;

	if( sqrtXUpper * sqrtXUpper == x )
	{
		return sqrtXUpper;
	}
	else
	{
		return sqrtXLower;
	}	
}

// static
bool Arithmetic::inRangeExclusive( float x, float lo, float hi )
{
	return( lo <= x && x < hi );
}

// static
bool Arithmetic::inRangeInclusive( float x, float lo, float hi )
{
	return( lo <= x && x <= hi );
}

//////////////////////////////////////////////////////////////////////////
// Private
//////////////////////////////////////////////////////////////////////////

// static
const float Arithmetic::s_fReciprocalLog2 = 1.f / logf( 2.f );

// static
const double Arithmetic::s_dDoubleMagicRoundEpsilon = 0.5 - 1.4e-11;

// static
const double Arithmetic::s_dDoubleMagic = 6755399441055744.0;
