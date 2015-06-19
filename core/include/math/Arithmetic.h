#pragma once

#include <common/BasicTypes.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector2i.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector3i.h>
#include <vecmath/Vector4f.h>
#include <vecmath/Vector4i.h>

class Arithmetic
{
public:

	// in C++, % is the remaidner
	// this gives the modulus
	// e.g.
	// -1 % 10 = -1
	// mod( -1, 10 ) = 9
	static int mod( int x, int n );
	static Vector2i mod( const Vector2i& v, const Vector2i& n );
	static Vector3i mod( const Vector3i& v, const Vector3i& n );
	static Vector4i mod( const Vector4i& v, const Vector4i& n );

	static int sign( int x );
	static int sign( float x );
	static int sign( double x );
	static Vector2i sign( const Vector2f& v );
	static Vector3i sign( const Vector3f& v );
	static Vector4i sign( const Vector4f& v );

	static bool sameSign( float x, float y );

	// static_cast< float >( numerator ) / denominator
	static float divideIntsToFloat( int numerator, int denominator );
	// round( divideIntsToFloat( numerator, denominator ) );
	static int divideIntsToFloatAndRound( int numerator, int denominator );
	// 100.0f * divideIntsToFloat( numerator, denominator ) )
	static float percentage( int numerator, int denominator );

	// given an array of length "arraySize", and bins of size "binSize"
	// computes the minimum number of bins needed to cover all arraySize elements.
	//   - The last bin may not be full
	//   - Simply divides them as floats and takes the ceil, returning it as an integer
	static int numBins( int arraySize, int binSize );

	// for x >= 1, returns true if x is a power of 2
	static bool isPowerOfTwo( int x );
	
	static int roundToInt( float x );
	static int floatToInt( float x ); // same as a static cast (truncates toward 0)
	static int floorToInt( float x ); // same as floor(x), followed by static cast (actually does floor)
	static int ceilToInt( float x );

	static int roundToInt( double x );
	static int doubleToInt( double x ); // same as a static cast (truncates toward 0)
	static int floorToInt( double x ); // same as floor(x), followed by static cast (actually does floor)
	static int ceilToInt( double x );

	static Vector2f floor( const Vector2f& v );
	static Vector2f ceil( const Vector2f& v );
	static Vector2i roundToInt( const Vector2f& v );
	static Vector2i floorToInt( const Vector2f& v );
	static Vector2i ceilToInt( const Vector2f& v );

	static Vector3f floor( const Vector3f& v );
	static Vector3f ceil( const Vector3f& v );
	static Vector3i roundToInt( const Vector3f& v );
	static Vector3i floorToInt( const Vector3f& v );
	static Vector3i ceilToInt( const Vector3f& v );

	static Vector4f floor( const Vector4f& v );
	static Vector4f ceil( const Vector4f& v );
	static Vector4i roundToInt( const Vector4f& v );
	static Vector4i floorToInt( const Vector4f& v );
	static Vector4i ceilToInt( const Vector4f& v );

    // Integer log2, computed with shifts and rounding down.
    // log2( 1 ) = 0
    // log2( 2 ) = 1
    // log2( 3 ) = 1
    // log2( 8 ) = 3
    // log2( 9 ) = 3
    // Returns -1 if x <= 0.
    static int log2( int x );

    // Fast float log2.
	static float log2( float x );
    // Fast approximate floor(log2(x))
	static int log2ToInt( float x );
	
	// From: http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
	// returns 0 if v is 0 (add v += ( v == 0 )) to return 1 in that case
	static uint32_t roundUpToNearestPowerOfTwo( uint32_t v );

	// Rounds x up to the nearest multiple of <n>.
    // If x is already a multile of <n>, then returns x.
    // Still works when x is zero or negative.
	static int roundUpToNearestMultipleOf4( int x );
	static int roundUpToNearestMultipleOf8( int x );
	static int roundUpToNearestMultipleOf16( int x );
	static int roundUpToNearestMultipleOf128( int x );
	static int roundUpToNearestMultipleOf256( int x );

	// finds y where y is the next perfect square greater than or equal to x
	// and optionally returns the square root
	static int findNextPerfectSquare( int x );
	static int findNextPerfectSquare( int x, int& sqrtOut );

	// returns true if x is a perfect square
	// optionally returning the square root
	static bool isPerfectSquare( int x );
	static bool isPerfectSquare( int x, int& sqrtOut );

	static int integerSquareRoot( int x );

	// returns true if lo <= x < hi
	static bool inRangeExclusive( float x, float lo, float hi );

	// returns true if lo <= x <= hi
	static bool inRangeInclusive( float x, float lo, float hi );

private:

	// almost .5f = .5f - 1e^(number of exp bit)
	static const double s_dDoubleMagicRoundEpsilon;
	static const double s_dDoubleMagic;

	static const float s_fReciprocalLog2;
};
