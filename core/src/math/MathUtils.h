#pragma once

#include "common/BasicTypes.h"
#include "math/Arithmetic.h"

// TODO: merge with Arithmetic

class Box3f;
class Box3i;
class Range1f;
class Range1i;
class Rect2f;
class Rect2i;
class Vector2f;
class Vector3f;
class Vector4f;
class Vector2i;
class Vector3i;
class Vector4i;

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

// Clamps x to [lo, hi).
int clampToRangeExclusive( int x, int lo, int hi );

// Clamps x to [lo, hi].
float clampToRangeInclusive( float x, float lo, float hi );

// Clamps x to [lo, hi].
double clampToRangeInclusive( double x, double lo, double hi );

// Clamp x to range (exclusive).
int clamp( int x, const Range1i& range );

// Clamp x to range (inclusive).
float clamp( float x, const Range1f& range );

// Clamps v to rect (exclusive).
Vector2i clamp( const Vector2i& v, const Rect2i& rect );

// Clamps v to rect (inclusive).
Vector2f clamp( const Vector2f& v, const Rect2f& rect );

// Clamps v to box (exclusive).
Vector3i clamp( const Vector3i& v, const Box3i& box );

// Clamps v to box (inclusive).
Vector3f clamp( const Vector3f& v, const Box3f& box );

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
T lerp( T x, T y, float t );

template< typename T >
T lerp( T x, T y, double t );

// Lerp between range.left() and range.right() by t.
float lerp( const Range1f& range, float t );

// Lerp between range.left() and range.right() by t.
float lerp( const Range1i& range, float t );

// TODO(jiawen): can make fraction templatized on output and input:
// fraction<double>( T x, Range< T > range );
// Returns what fraction of the way x is between range.left() and
// range.right(). Equal to (x - range.left()) / range.size. If x is in the
// range, then returns a value in [0, 1]. x outside the range will *not* be
// clamped.
float fraction( float x, const Range1f& range );

// Returns what fraction of the way x is between range.left() and
// range.right(). Equal to (x - range.left()) / range.size. If x is in the
// range, then returns a value in [0, 1]. x outside the range will *not* be
// clamped.
float fraction( int x, const Range1i& range );

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

float rescale( float x, const Range1f& src, const Range1f& dst );

int rescale( float x, const Range1f& src, const Range1i& dst );

float rescale( int x, const Range1i& src, const Range1f& dst );

int rescale( int x, const Range1i& src, const Range1i& dst );

// ----- Cubic spline interpolation -----

// TODO: this is Catmull-Rom?
// Look at 6.839 notes.
template< typename T >
T cubicInterpolate( const T& p0, const T& p1, const T& p2, const T& p3,
    float t );

// ----- Misc -----

float distanceSquared( float x0, float y0, float x1, float y1 );

// Evaluates a continuous normal distribution with mean u and standard
// deviation sigma.
float gaussian( float x, float u, float sigma );

// 1/x, returns 0 if x=0
float oo_0( float x );
double oo_0( double x );

} } } // math, core, libcgt

#include "MathUtils.inl"
