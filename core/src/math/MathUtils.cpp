#include "math/MathUtils.h"

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <limits>

#include "math/Arithmetic.h"

// In Visual Studio, if _MATH_DEFINES_DEFINED is defined
// then M_PI, etc are defined, so just copy them
#ifdef _MATH_DEFINES_DEFINED

const float MathUtils::E = M_E;
const float MathUtils::PI = M_PI;
const float MathUtils::HALF_PI = M_PI_2;
const float MathUtils::QUARTER_PI = M_PI_4;
const float MathUtils::TWO_PI = 2.0f * M_PI;

#else

const float MathUtils::E = 2.71828182845904523536f;
const float MathUtils::PI = 3.14159265358979323846f;
const float MathUtils::HALF_PI = 1.57079632679489661923f;
const float MathUtils::QUARTER_PI = 0.78539816339744830962f;
const float MathUtils::TWO_PI = 2.0f * MathUtils::PI;

#endif

// static
const float MathUtils::NEGATIVE_INFINITY = -std::numeric_limits< float >::infinity();

// static
const float MathUtils::POSITIVE_INFINITY = std::numeric_limits< float >::infinity();

// static
float MathUtils::cot( float x )
{
	return 1.f / tanf( x );
}

// static
float MathUtils::asinh( float x )
{
    return log(x + sqrt(x * x + 1.f));
}

// static
int MathUtils::sign( float f )
{
	if( f < 0 )
	{
		return -1;
	}
	else if( f > 0 )
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

// static
bool MathUtils::sameSign( float x, float y )
{	
	return sign( x ) == sign( y );
}

//static
bool MathUtils::isNumber( float x )
{
	// See: http://www.johndcook.com/IEEE_exceptions_in_cpp.html
	// returns false if x is NaN
	return( x == x );
}

// static
bool MathUtils::isFinite( float x )
{
	// See: http://www.johndcook.com/IEEE_exceptions_in_cpp.html
	return( x <= FLT_MAX && x >= -FLT_MAX );
}

// static
float MathUtils::degreesToRadians( float degrees )
{
	return static_cast< float >( degrees * MathUtils::PI / 180.0f );
}

// static
double MathUtils::degreesToRadians( double degrees )
{
	return( degrees * MathUtils::PI / 180.0 );
}

// static
float MathUtils::radiansToDegrees( float radians )
{
	return static_cast< float >( radians * 180.0f / MathUtils::PI );
}

// static
double MathUtils::radiansToDegrees( double radians )
{
	return( radians * 180.0 / MathUtils::PI );
}

// static
int MathUtils::clampToRangeExclusive( int x, int lo, int hi )
{
	if( x >= hi )
	{
		x = hi - 1;
	}
	if( x < lo )
	{
		x = lo;
	}

	return x;
}

// static
int MathUtils::clampToRangeInclusive( int x, int lo, int hi )
{
	if( x > hi )
	{
		x = hi;
	}
	if( x < lo )
	{
		x = lo;
	}

	return x;
}

// static
float MathUtils::clampToRange( float x, float lo, float hi )
{
	if( x > hi )
	{
		x = hi;
	}
	if( x < lo )
	{
		x = lo;
	}
	
	return x;
}

// static
double MathUtils::clampToRange( double x, double lo, double hi )
{
	if( x > hi )
	{
		x = hi;
	}
	if( x < lo )
	{
		x = lo;
	}

	return x;
}

// static
sbyte MathUtils::floatToByteSignedNormalized( float f )
{
	return static_cast< sbyte >( f * 127 );
}

// static
float MathUtils::signedByteToFloatNormalized( sbyte sb )
{
	return( sb / 127.f );
}

// static
void MathUtils::rescaleRangeToScaleOffset( float inputMin, float inputMax,
	float outputMin, float outputMax,
	float& scale, float& offset )
{
	float inputRange = inputMax - inputMin;
	float outputRange = outputMax - outputMin;

	// y = outputMin + [ ( x - inputMin ) / inputRange ] * outputRange
	//   = outputMin + ( x * outputRange / inputRange ) - ( inputMin * outputRange / inputRange )
	//
	// -->
	//
	// scale = outputRange / inputRange
	// offset = outputMin - inputMin * outputRange / inputRange

	scale = outputRange / inputRange;
	offset = outputMin - inputMin * scale;
}

// static
float MathUtils::rescaleFloatToFloat( float x,
	float inputMin, float inputMax,
	float outputMin, float outputMax )
{
	float inputRange = inputMax - inputMin;
	float outputRange = outputMax - outputMin;

	float fraction = ( x - inputMin ) / inputRange;

	return( outputMin + fraction * outputRange );
}

// static
int MathUtils::rescaleFloatToInt( float x,
	float fMin, float fMax,
	int iMin, int iMax )
{
	float fraction = ( x - fMin ) / ( fMax - fMin );
	return( iMin + Arithmetic::roundToInt( fraction * ( iMax - iMin ) ) );
}

// static
float MathUtils::rescaleIntToFloat( int x,
	int iMin, int iMax,
	float fMin, float fMax )
{
	int inputRange = iMax - iMin;
	float outputRange = fMax - fMin;

	float fraction = static_cast< float >( x - iMin ) / inputRange;
	return( fMin + fraction * outputRange );
}

// static
int MathUtils::rescaleIntToInt( int x,
	int inMin, int inMax,
	int outMin, int outMax )
{
	int inputRange = inMax - inMin;
	int outputRange = outMax - outMin;

	float fraction = static_cast< float >( x - inMin )  / inputRange;
	return Arithmetic::roundToInt( outMin + fraction * outputRange );
}

// static
float MathUtils::cubicInterpolate( float p0, float p1, float p2, float p3, float t )
{
	// geometric construction:
	//            t
	//   (t+1)/2     t/2
	// t+1        t	        t-1

	// bottom level
	float p0p1 = lerp( p0, p1, t + 1 );
	float p1p2 = lerp( p1, p2, t );
	float p2p3 = lerp( p2, p3, t - 1 );

	// middle level
	float p0p1_p1p2 = lerp( p0p1, p1p2, 0.5f * ( t + 1 ) );
	float p1p2_p2p3 = lerp( p1p2, p2p3, 0.5f * t );

	// top level
	return lerp( p0p1_p1p2, p1p2_p2p3, t );
}

// static
float MathUtils::distanceSquared( float x0, float y0, float x1, float y1 )
{
	float dx = x1 - x0;
	float dy = y1 - y0;

	return( dx * dx + dy * dy );
}

// static
float MathUtils::gaussianWeight( float r, float sigma )
{
	return exp( -( r * r ) / ( 2.f * sigma * sigma ) );
}
