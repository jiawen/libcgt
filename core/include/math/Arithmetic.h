#ifndef ARITHMETIC_H
#define ARITHMETIC_H

#include <common/BasicTypes.h>

class Arithmetic
{
public:

	// in C++, % is the remaidner
	// this gives the modulus
	// e.g.
	// -1 % 10 = -1
	// mod( -1, 10 ) = 9
	static int mod( int x, int N );

	static float divideIntsToFloat( int numerator, int denominator );
	static int divideIntsToFloatAndRound( int numerator, int denominator );

	static bool isPowerOfTwo( int x );
	
	static int roundToInt( double val );
	static int floatToInt( double val );
	static int floorToInt( double val );
	static int ceilToInt( double val );

	static float log2( float x );
	static int log2ToInt( float v );
	static uint roundUpToNearestPowerOfTwo( uint v );

	// finds y where y is the next perfect square greater than or equal to x
	// if sqrtOut != NULL, returns the square root of y in sqrtOut
	static int findNextPerfectSquare( int x, int* sqrtOut = NULL );

	// returns true if x is a perfect square
	// if it is, then pSqrt contains the square root
	// pass in NULL for testing only
	static bool isPerfectSquare( int x, int* sqrtOut = NULL );

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

#endif // ARITHMETIC_H
