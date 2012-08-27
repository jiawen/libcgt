#pragma once

#include "common/BasicTypes.h"

class MathUtils
{
public:

	static const float E;
	static const float PI;
	static const float HALF_PI;
	static const float QUARTER_PI;
	static const float TWO_PI;

	static const float NEGATIVE_INFINITY;
	static const float POSITIVE_INFINITY;

	static float cot( float x );
    static float asinh(float x);

	static int sign( float f );
	static bool sameSign( float x, float y );

	// returns true if x is not NaN
	// -inf and +inf *are* considered numbers
	static bool isNumber( float x );

	// returns true if x is not one of: {NaN, -inf, +inf}
	static bool isFinite( float x );

	static float degreesToRadians( float degrees );
	static double degreesToRadians( double degrees );

	static float radiansToDegrees( float radians );
	static double radiansToDegrees( double radians );

	// clamps x to [lo, hi)
	static int clampToRangeExclusive( int x, int lo, int hi );

	// clamps x to [lo, hi]
	static int clampToRangeInclusive( int x, int lo, int hi );

	// clamps x to between min (inclusive) and max (inclusive)
	static float clampToRange( float x, float lo, float hi );
	static double clampToRange( double x, double lo, double hi );

	// converts a float in [-1,1] to
	// a signed byte in [-127,127]
	// the behavior for f outside [-1,1] is undefined
	static sbyte floatToByteSignedNormalized( float f );

	// converts a signed byte in [-127,127] to
	// a [snorm] float in [-1,1]
	static float signedByteToFloatNormalized( sbyte sb );

	// given an input range [inputMin, inputMax]
	// and an output range [outputMin, outputMax]
	//
	// returns the parameters scale and offset such that
	// for x \in [inputMin, inputMax]
	// scale * x + offset is in the range [outputMin,outputMax]	
	static void rescaleRangeToScaleOffset( float inputMin, float inputMax,
		float outputMin, float outputMax,
		float& scale, float& offset );

	static float rescaleFloatToFloat( float x,
		float inputMin, float inputMax,
		float outputMin, float outputMax );

	// rescales a float range to an int range, with rounding
	// iMax is inclusive
	static int rescaleFloatToInt( float x,
		float fMin, float fMax,
		int iMin, int iMax );

	// iMax is inclusive
	static float rescaleIntToFloat( int x,
		int iMin, int iMax,
		float fMin, float fMax );

	// rescale an int range with rounding,
	// inMax and out Max are inclusive
	static int rescaleIntToInt( int x,
		int inMin, int inMax,
		int outMin, int outMax );

	template< typename T >
	static T lerp( const T& x, const T& y, float t );

	static float cubicInterpolate( float p0, float p1, float p2, float p3, float t );

	static float distanceSquared( float x0, float y0, float x1, float y1 );

	static float gaussianWeight( float r, float sigma );

	// 1/x, returns 0 if x=0
	static inline float oo_0( float x );
	static inline double oo_0( double x );

private:

};

#include "MathUtils.inl"