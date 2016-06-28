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

namespace libcgt { namespace core { namespace math {

// ----- Constants -----
extern const float E;
extern const float PI;
extern const float HALF_PI;
extern const float QUARTER_PI;
extern const float TWO_PI;

extern const float PHI; // The golden ratio ( 1 + sqrt( 5 ) ) / 2.

extern const float NEGATIVE_INFINITY;
extern const float POSITIVE_INFINITY;

// ----- Numbers -----

// Returns true if x is not NaN.
// -inf and +inf *are* considered numbers
bool isNumber( float x );

// Returns true if x is *not* one of: {NaN, -inf, +inf}.
bool isFinite( float x );

// ----- Trigonometry -----

float asinh( float x );
double asinh( double x );

float cot( float x );
double cot( double x );

float degreesToRadians( float degrees );
double degreesToRadians( double degrees );

float radiansToDegrees( float radians );
double radiansToDegrees( double radians );

// ----- Clamping -----

// TODO: these need to be changed to [origin, size) as well
// clampToRectangleExclusive and clampToBoxExclusive depends on these
// clamps x to [lo, hi)
int clampToRangeExclusive( int x, int lo, int hi );

// clamps x to [lo, hi]
int clampToRangeInclusive( int x, int lo, int hi );

// clamps x to between min (inclusive) and max (inclusive)
template< typename T >
T clampToRange( const T& x, const T& lo, const T& hi );

// clamps v to rect (exclusive)
Vector2i clampToRectangleExclusive( const Vector2i& v, const Rect2i& rect );

// clamps v to rect (inclusive)
Vector2f clampToRectangle( const Vector2f& v, const Rect2f& rect );

// clamps v to box (exclusive)
Vector3i clampToBoxExclusive( const Vector3i& v, const Box3i& box );

// clamps v to box (inclusive)
Vector3f clampToBox( const Vector3f& v, const Box3f& box );

// ----- Absolute value of all components of a short vector -----
Vector2f abs( const Vector2f& v );
Vector3f abs( const Vector3f& v );
Vector4f abs( const Vector4f& v );

Vector2i abs( const Vector2i& v );
Vector3i abs( const Vector3i& v );
Vector4i abs( const Vector4i& v );

// ----- Product of all components of a short vector -----
float product( const Vector2f& v );
float product( const Vector3f& v );
float product( const Vector4f& v );

int product( const Vector2i& v );
int product( const Vector3i& v );
int product( const Vector4i& v );

// ----- min/max of all components of a short vector -----
float minimum( const Vector2f& v );
float minimum( const Vector3f& v );
float minimum( const Vector4f& v );

int minimum( const Vector2i& v );
int minimum( const Vector3i& v );
int minimum( const Vector4i& v );

float maximum( const Vector2f& v );
float maximum( const Vector3f& v );
float maximum( const Vector4f& v );

int maximum( const Vector2i& v );
int maximum( const Vector3i& v );
int maximum( const Vector4i& v );

// ----- component-wise min/max of short vectors -----
Vector2f minimum( const Vector2f& v0, const Vector2f& v1 );
Vector3f minimum( const Vector3f& v0, const Vector3f& v1 );
Vector4f minimum( const Vector4f& v0, const Vector4f& v1 );

Vector2i minimum( const Vector2i& v0, const Vector2i& v1 );
Vector3i minimum( const Vector3i& v0, const Vector3i& v1 );
Vector4i minimum( const Vector4i& v0, const Vector4i& v1 );

Vector2f maximum( const Vector2f& v0, const Vector2f& v1 );
Vector3f maximum( const Vector3f& v0, const Vector3f& v1 );
Vector4f maximum( const Vector4f& v0, const Vector4f& v1 );

Vector2i maximum( const Vector2i& v0, const Vector2i& v1 );
Vector3i maximum( const Vector3i& v0, const Vector3i& v1 );
Vector4i maximum( const Vector4i& v0, const Vector4i& v1 );

// ----- Linear interpolation -----

template< typename T >
T lerp( const T& x, const T& y, float t );

Vector2f lerp( const Vector2i& v0, const Vector2i& v1, float alpha );
Vector3f lerp( const Vector3i& v0, const Vector3i& v1, float alpha );
Vector4f lerp( const Vector4i& v0, const Vector4i& v1, float alpha );

// TODO(jiawen): Move this into RangeUtils.

// given an input range [inputMin, inputMax]
// and an output range [outputMin, outputMax]
//
// Computes the parameters scale and offset such that:
// for x \in [inputMin, inputMax]
// scale * x + offset is in the range [outputMin, outputMax].
// Call this transformBetween() just like for Rect2f.
void rescaleRangeToScaleOffset( float inputMin, float inputMax,
    float outputMin, float outputMax,
    float& scale, float& offset );

float rescaleFloatToFloat( float x,
    float inputMin, float inputMax,
    float outputMin, float outputMax );

// rescales a float range to an int range, with rounding
// iMax is inclusive
int rescaleFloatToInt( float x,
    float fMin, float fMax,
    int iMin, int iMax );

// iMax is inclusive
float rescaleIntToFloat( int x,
    int iMin, int iMax,
    float fMin, float fMax );

// rescale an int range with rounding,
// inMax and outMax are inclusive
int rescaleIntToInt( int x,
    int inMin, int inMax,
    int outMin, int outMax );

// ----- Cubic spline interpolation -----

// TODO: this is Catmull-Rom?
// Look at 6.839 notes.
template< typename T >
T cubicInterpolate( const T& p0, const T& p1, const T& p2, const T& p3, float t );

// ----- Misc -----

float distanceSquared( float x0, float y0, float x1, float y1 );

// Evaluates a continuous normal distribution with mean u and standard
// deviation sigma.
float gaussian( float x, float u, float sigma );

// 1/x, returns 0 if x=0
inline float oo_0( float x );
inline double oo_0( double x );

} } } // math, core, libcgt

#include "MathUtils.inl"
