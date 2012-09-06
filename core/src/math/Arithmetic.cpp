#include <cassert>
#include <cmath>

#include "math/Arithmetic.h"

// static
int Arithmetic::mod( int x, int N )
{
	return ( ( x % N ) + N ) % N;
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
int Arithmetic::roundToInt( float val )
{
	return static_cast< int >( val + 0.5f );
}

// static
int Arithmetic::floatToInt( float val )
{
	return static_cast< int >( val );
}

// static
int Arithmetic::floorToInt( float val )
{
	return static_cast< int >( floor( val ) );
}

// static
int Arithmetic::ceilToInt( float val )
{
	return static_cast< int >( ceil( val ) );
}

// static
int Arithmetic::roundToInt( double val )
{
	// 2^52 * 1.5, uses limited precision to floor
	val	= val + Arithmetic::s_dDoubleMagic;
	return( ( int* ) &val )[0];
}

// static
int Arithmetic::doubleToInt( double val )
{
	return( ( val < 0 ) ?
		Arithmetic::roundToInt( val + s_dDoubleMagicRoundEpsilon ) :
		Arithmetic::roundToInt( val - s_dDoubleMagicRoundEpsilon ) );
}

// static
int Arithmetic::floorToInt( double val )
{
	return Arithmetic::roundToInt( val - s_dDoubleMagicRoundEpsilon );
}

// static
int Arithmetic::ceilToInt( double val )
{
	return Arithmetic::roundToInt( val + s_dDoubleMagicRoundEpsilon );
}

// static
float Arithmetic::log2( float x )
{	
	return( logf( x ) * Arithmetic::s_fReciprocalLog2 );
}

// static
int Arithmetic::log2ToInt( float v )
{
	return( ( *( int* )( &v ) ) >> 23 ) - 127;
}

// static
uint Arithmetic::roundUpToNearestPowerOfTwo( uint v )
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
