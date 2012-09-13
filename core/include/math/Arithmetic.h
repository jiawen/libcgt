#pragma once

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

	static int sign( int x );
	static int sign( float x );
	static int sign( double x );

	static float divideIntsToFloat( int numerator, int denominator );
	static int divideIntsToFloatAndRound( int numerator, int denominator );

	// given an array of length "arraySize", and bins of size "binSize"
	// computes the minimum number of bins needed to cover all arraySize elements.
	//   - The last bin may not be full
	//   - Simply divides them as floats and takes the ceil, returning it as an integer
	static int numBins( int arraySize, int binSize );

	static bool isPowerOfTwo( int x );
	
	static int roundToInt( float val );
	static int floatToInt( float val ); // same as a static cast
	static int floorToInt( float val ); // same as floor(val), followed by static cast
	static int ceilToInt( float val );

	static int roundToInt( double val );
	static int doubleToInt( double val ); // same as a static cast
	static int floorToInt( double val ); // same as floor(val), followed by static cast
	static int ceilToInt( double val );

	static float log2( float x );
	static int log2ToInt( float v );
	
	// From: http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
	// returns 0 if v is 0 (add v += ( v == 0 )) to return 1 in that case
	static uint roundUpToNearestPowerOfTwo( uint v );

	// leaves x alone if it's already a multiple
	static int roundUpToNearestMultipleOf4( int x );
	static int roundUpToNearestMultipleOf8( int x );
	static int roundUpToNearestMultipleOf16( int x );
	static int roundUpToNearestMultipleOf128( int x );
	static int roundUpToNearestMultipleOf256( int x );

	// finds y where y is the next perfect square greater than or equal to x
	// and optionally reqturns the square root
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
