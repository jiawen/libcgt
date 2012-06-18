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

	static float cot( float x );
    static float asinh(float x);

	static int sign( float f );
	static bool sameSign( float x, float y );

	static float degreesToRadians( float degrees );
	static double degreesToRadians( double degrees );

	static float radiansToDegrees( float radians );
	static double radiansToDegrees( double radians );

	// clamps x to [lo, hi)
	static int clampToRangeExclusive( int x, int lo, int hi );

	// clamps x to [lo, hi]
	static int clampToRangeInclusive( int x, int lo, int hi );

	// clamps x to between min (inclusive) and max (includisve)
	static float clampToRange( float x, float lo, float hi );
	static double clampToRange( double x, double lo, double hi );

	// converts a float in [-1,1] to
	// a signed byte in [-127,127]
	// the behavior for f outside [-1,1] is undefined
	static sbyte floatToByteSignedNormalized( float f );

	// converts a signed byte in [-127,127] to
	// a [snorm] float in [-1,1]
	static float signedByteToFloatNormalized( sbyte sb );

	static float rescaleFloatToFloat( float value,
		float inputMin, float inputMax,
		float outputMin, float outputMax );

	static int rescaleFloatToInt( float value,
		float fMin, float fMax,
		int iMin, int iMax );

	static float rescaleIntToFloat( int value,
		int iMin, int iMax,
		float fMin, float fMax );

	static int rescaleIntToInt( int value,
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