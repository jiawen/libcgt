#pragma once

#include "common/BasicTypes.h"

// TODO: turn into pure functions
// TODO: merge with Arithmetic

class Vector2f;
class Vector3f;
class Vector4f;
class Vector2i;
class Vector3i;
class Vector4i;

class Rect2f;
class Rect2i;
class Box3f;
class Box3i;

class MathUtils
{
public:


    // ----- Constants -----
    static const float E;
    static const float PI;
    static const float HALF_PI;
    static const float QUARTER_PI;
    static const float TWO_PI;

    static const float PHI; // golden ratio ( 1 + sqrt( 5 ) ) / 2

    static const float NEGATIVE_INFINITY;
    static const float POSITIVE_INFINITY;

    // ----- Numbers -----

    // returns true if x is not NaN
    // -inf and +inf *are* considered numbers
    static bool isNumber( float x );

    // returns true if x is not one of: {NaN, -inf, +inf}
    static bool isFinite( float x );

    // ----- Trigonometry -----

    static float cot( float x );
    static float asinh(float x);

    static float degreesToRadians( float degrees );
    static double degreesToRadians( double degrees );

    static float radiansToDegrees( float radians );
    static double radiansToDegrees( double radians );

    // ----- Clamping -----

    // TODO: these need to be changed to [origin, size) as well
    // clampToRectangleExclusive and clampToBoxExclusive depends on these
    // clamps x to [lo, hi)
    static int clampToRangeExclusive( int x, int lo, int hi );

    // clamps x to [lo, hi]
    static int clampToRangeInclusive( int x, int lo, int hi );

    // clamps x to between min (inclusive) and max (inclusive)
    template< typename T >
    static T clampToRange( const T& x, const T& lo, const T& hi );

    // clamps v to rect (exclusive)
    static Vector2i clampToRectangleExclusive( int x, int y, int left, int bottom, int width, int height );
    static Vector2i clampToRectangleExclusive( const Vector2i& v, const Vector2i& origin, const Vector2i& size );
    static Vector2i clampToRectangleExclusive( const Vector2i& v, const Vector2i& size );
    static Vector2i clampToRectangleExclusive( const Vector2i& v, const Rect2i& rect );

    // clamps v to rect (inclusive)
    static Vector2f clampToRectangle( const Vector2f& v, const Rect2f& rect );

    // clamps v to box (exclusive)
    static Vector3i clampToBoxExclusive( int x, int y, int z, int left, int bottom, int back, int width, int height, int depth );
    static Vector3i clampToBoxExclusive( const Vector3i& v, const Vector3i& origin, const Vector3i& size );
    static Vector3i clampToBoxExclusive( const Vector3i& v, const Vector3i& size );
    static Vector3i clampToBoxExclusive( const Vector3i& v, const Box3i& box );

    // clamps v to box (inclusive)
    static Vector3f clampToBox( const Vector3f& v, const Box3f& box );

    // ----- Absolute value of all components of a short vector -----
    static Vector2f abs( const Vector2f& v );
    static Vector3f abs( const Vector3f& v );
    static Vector4f abs( const Vector4f& v );

    static Vector2i abs( const Vector2i& v );
    static Vector3i abs( const Vector3i& v );
    static Vector4i abs( const Vector4i& v );

    // ----- Product of all components of a short vector -----
    static float product( const Vector2f& v );
    static float product( const Vector3f& v );
    static float product( const Vector4f& v );

    static int product( const Vector2i& v );
    static int product( const Vector3i& v );
    static int product( const Vector4i& v );

    // ----- min/max of all components of a short vector -----
    static float minimum( const Vector2f& v );
    static float minimum( const Vector3f& v );
    static float minimum( const Vector4f& v );

    static int minimum( const Vector2i& v );
    static int minimum( const Vector3i& v );
    static int minimum( const Vector4i& v );

    static float maximum( const Vector2f& v );
    static float maximum( const Vector3f& v );
    static float maximum( const Vector4f& v );

    static int maximum( const Vector2i& v );
    static int maximum( const Vector3i& v );
    static int maximum( const Vector4i& v );

    // ----- component-wise min/max of short vectors -----
    static Vector2f minimum( const Vector2f& v0, const Vector2f& v1 );
    static Vector3f minimum( const Vector3f& v0, const Vector3f& v1 );
    static Vector4f minimum( const Vector4f& v0, const Vector4f& v1 );

    static Vector2i minimum( const Vector2i& v0, const Vector2i& v1 );
    static Vector3i minimum( const Vector3i& v0, const Vector3i& v1 );
    static Vector4i minimum( const Vector4i& v0, const Vector4i& v1 );

    static Vector2f maximum( const Vector2f& v0, const Vector2f& v1 );
    static Vector3f maximum( const Vector3f& v0, const Vector3f& v1 );
    static Vector4f maximum( const Vector4f& v0, const Vector4f& v1 );

    static Vector2i maximum( const Vector2i& v0, const Vector2i& v1 );
    static Vector3i maximum( const Vector3i& v0, const Vector3i& v1 );
    static Vector4i maximum( const Vector4i& v0, const Vector4i& v1 );

    // ----- Linear interpolation -----

    template< typename T >
    static T lerp( const T& x, const T& y, float t );

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
    // inMax and outMax are inclusive
    static int rescaleIntToInt( int x,
        int inMin, int inMax,
        int outMin, int outMax );

    // ----- Cubic spline interpolation -----

    // TODO: this is Catmull-Rom?
    // Look at 6.839 notes.
    template< typename T >
    static T cubicInterpolate( const T& p0, const T& p1, const T& p2, const T& p3, float t );

    // ----- Misc -----

    static float distanceSquared( float x0, float y0, float x1, float y1 );

    // evaluates a continuous normal distribution with mean u and standard deviation sigma
    static float gaussian( float x, float u, float sigma );

    // 1/x, returns 0 if x=0
    static inline float oo_0( float x );
    static inline double oo_0( double x );

private:

};

#include "MathUtils.inl"