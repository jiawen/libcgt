#pragma once

#include "libcgt/core/common/BasicTypes.h"
#include "libcgt/core/vecmath/Vector2f.h"
#include "libcgt/core/vecmath/Vector2i.h"
#include "libcgt/core/vecmath/Vector3f.h"
#include "libcgt/core/vecmath/Vector3i.h"
#include "libcgt/core/vecmath/Vector4f.h"
#include "libcgt/core/vecmath/Vector4i.h"

namespace libcgt { namespace core { namespace math {

// Compute the "modulus of division"
// In C++, % is the remainder operator, where for negative arguments:
// -1 % 10 = -1
// This function always computes a positive result:
// mod( -1, 10 ) = 9
int mod( int x, int n );
Vector2i mod( const Vector2i& v, const Vector2i& n );
Vector3i mod( const Vector3i& v, const Vector3i& n );
Vector4i mod( const Vector4i& v, const Vector4i& n );

// Compute the sign of the argument.
int sign( int x );
int sign( float x );
int sign( double x );
Vector2i sign( const Vector2f& v );
Vector3i sign( const Vector3f& v );
Vector4i sign( const Vector4f& v );

bool sameSign( float x, float y );

// static_cast< float >( numerator ) / denominator
float divideIntsToFloat( int numerator, int denominator );
// round( divideIntsToFloat( numerator, denominator ) );
int divideIntsToFloatAndRound( int numerator, int denominator );
// 100.0f * divideIntsToFloat( numerator, denominator ) )
float percentage( int numerator, int denominator );

// given an array of length "arraySize", and bins of size "binSize"
// computes the minimum number of bins needed to cover all arraySize elements.
//   - The last bin may not be full
//   - Simply divides them as floats and takes the ceil, returning it as an integer
int numBins( int arraySize, int binSize );

// for x >= 1, returns true if x is a power of 2
bool isPowerOfTwo( int x );

int roundToInt( float x );
int floatToInt( float x ); // same as a cast (truncates toward 0)
int floorToInt( float x ); // same as floor(x), followed by cast (actually does floor)
int ceilToInt( float x );

int roundToInt( double x );
int doubleToInt( double x ); // same as a cast (truncates toward 0)
int floorToInt( double x ); // same as floor(x), followed by cast (actually does floor)
int ceilToInt( double x );

Vector2f floor( const Vector2f& v );
Vector2f ceil( const Vector2f& v );
Vector2i roundToInt( const Vector2f& v );
Vector2i floorToInt( const Vector2f& v );
Vector2i ceilToInt( const Vector2f& v );

Vector3f floor( const Vector3f& v );
Vector3f ceil( const Vector3f& v );
Vector3i roundToInt( const Vector3f& v );
Vector3i floorToInt( const Vector3f& v );
Vector3i ceilToInt( const Vector3f& v );

Vector4f floor( const Vector4f& v );
Vector4f ceil( const Vector4f& v );
Vector4i roundToInt( const Vector4f& v );
Vector4i floorToInt( const Vector4f& v );
Vector4i ceilToInt( const Vector4f& v );

// Integer log2, computed with shifts and rounding down.
// log2( 1 ) = 0
// log2( 2 ) = 1
// log2( 3 ) = 1
// log2( 8 ) = 3
// log2( 9 ) = 3
// Returns -1 if x <= 0.
int log2( int x );

// Fast float log2.
float log2( float x );
// Fast approximate floor(log2(x))
int log2ToInt( float x );

// From: http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
// returns 0 if v is 0 (add v += ( v == 0 )) to return 1 in that case
uint32_t roundUpToNearestPowerOfTwo( uint32_t v );

// Rounds x up to the nearest multiple of <n>.
// If x is already a multile of <n>, then returns x.
// Still works when x is zero or negative.
int roundUpToNearestMultipleOf4( int x );
int roundUpToNearestMultipleOf8( int x );
int roundUpToNearestMultipleOf16( int x );
int roundUpToNearestMultipleOf128( int x );
int roundUpToNearestMultipleOf256( int x );

// finds y where y is the next perfect square greater than or equal to x
// and optionally returns the square root
int findNextPerfectSquare( int x );
int findNextPerfectSquare( int x, int& sqrtOut );

// returns true if x is a perfect square
// optionally returning the square root
bool isPerfectSquare( int x );
bool isPerfectSquare( int x, int& sqrtOut );

int integerSquareRoot( int x );

// TODO(jiawen): Move this into RangeUtils.

// returns true if lo <= x < hi
bool inRangeExclusive( float x, float lo, float hi );

// returns true if lo <= x <= hi
bool inRangeInclusive( float x, float lo, float hi );

} } } // math, core, libcgt
