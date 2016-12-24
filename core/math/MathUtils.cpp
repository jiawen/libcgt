#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <limits>

#include "libcgt/core/math/Arithmetic.h"
#include "libcgt/core/math/MathUtils.h"

#include "libcgt/core/vecmath/Rect2f.h"
#include "libcgt/core/vecmath/Rect2i.h"
#include "libcgt/core/vecmath/Box3f.h"
#include "libcgt/core/vecmath/Box3i.h"

using libcgt::core::math::roundToInt;

namespace libcgt { namespace core { namespace math {

const float E = static_cast< float >( M_E );
const float PI = static_cast< float >( M_PI );
const float HALF_PI = static_cast< float >( M_PI_2 );
const float QUARTER_PI = static_cast< float >( M_PI_4 );
const float TWO_PI = static_cast< float >( 2.0f * M_PI );

const float PHI = 0.5f * ( 1 + sqrt( 5.0f ) );

const float NEGATIVE_INFINITY = -std::numeric_limits< float >::infinity();

const float POSITIVE_INFINITY = std::numeric_limits< float >::infinity();

//static
bool isNumber( float x )
{
    // See: http://www.johndcook.com/IEEE_exceptions_in_cpp.html
    // returns false if x is NaN
    return( x == x );
}

bool isFinite( float x )
{
    // See: http://www.johndcook.com/IEEE_exceptions_in_cpp.html
    return( x <= FLT_MAX && x >= -FLT_MAX );
}

float asinh( float x )
{
    return log( x + sqrt( x * x + 1.0f ) );
}

double asinh( double x )
{
    return log( x + sqrt( x * x + 1.0 ) );
}

float cot( float x )
{
    return 1.0f / tan( x );
}

double cot( double x )
{
    return 1.0 / tan( x );
}

float degreesToRadians( float degrees )
{
    return static_cast< float >( degrees * PI / 180.0f );
}

double degreesToRadians( double degrees )
{
    return( degrees * PI / 180.0 );
}

float radiansToDegrees( float radians )
{
    return static_cast< float >( radians * 180.0f / PI );
}

double radiansToDegrees( double radians )
{
    return( radians * 180.0 / PI );
}

Vector2i clamp( const Vector2i& v, const Rect2i& rect )
{
    assert( rect.isStandard() );
    return
    {
        clampToRangeExclusive( v.x, rect.left(), rect.right() ),
        clampToRangeExclusive( v.y, rect.bottom(), rect.top() )
    };
}

Vector2f clamp( const Vector2f& v, const Rect2f& rect )
{
    assert( rect.isStandard() );
    return
    {
        clampToRangeInclusive( v.x, rect.left(), rect.right() ),
        clampToRangeInclusive( v.y, rect.bottom(), rect.top() )
    };
}

Vector3i clamp( const Vector3i& v, const Box3i& box )
{
    assert( box.isStandard() );
    return
    {
        clampToRangeExclusive( v.x, box.left(), box.right() ),
        clampToRangeExclusive( v.y, box.bottom(), box.top() ),
        clampToRangeExclusive( v.z, box.back(), box.front() )
    };
}

Vector3f clamp( const Vector3f& v, const Box3f& box )
{
    assert( box.isStandard() );
    return
    {
        clampToRangeInclusive( v.x, box.left(), box.right() ),
        clampToRangeInclusive( v.y, box.bottom(), box.top() ),
        clampToRangeInclusive( v.z, box.back(), box.front() )
    };
}

Vector2f abs( const Vector2f& v )
{
    return
    {
        std::abs( v.x ),
        std::abs( v.y )
    };
}

Vector3f abs( const Vector3f& v )
{
    return
    {
        std::abs( v.x ),
        std::abs( v.y ),
        std::abs( v.z )
    };
}

Vector4f abs( const Vector4f& v )
{
    return
    {
        std::abs( v.x ),
        std::abs( v.y ),
        std::abs( v.z ),
        std::abs( v.w )
    };
}

Vector2i abs( const Vector2i& v )
{
    return
    {
        std::abs( v.x ),
        std::abs( v.y )
    };
}

Vector3i abs( const Vector3i& v )
{
    return
    {
        std::abs( v.x ),
        std::abs( v.y ),
        std::abs( v.z )
    };
}

Vector4i abs( const Vector4i& v )
{
    return
    {
        std::abs( v.x ),
        std::abs( v.y ),
        std::abs( v.z ),
        std::abs( v.w )
    };
}

float product( const Vector2f& v )
{
    return v.x * v.y;
}

float product( const Vector3f& v )
{
    return v.x * v.y * v.z;
}

float product( const Vector4f& v )
{
    return v.x * v.y * v.z * v.w;
}

int product( const Vector2i& v )
{
    return v.x * v.y;
}

int product( const Vector3i& v )
{
    return v.x * v.y * v.z;
}

int product( const Vector4i& v )
{
    return v.x * v.y * v.z * v.w;
}

float minimum( const Vector2f& v )
{
    return std::min( v.x, v.y );
}

float minimum( const Vector3f& v )
{
    return std::min( v.x, std::min( v.y, v.z ) );
}

float minimum( const Vector4f& v )
{
    return std::min( v.x, std::min( v.y, std::min( v.z, v.w ) ) );
}

int minimum( const Vector2i& v )
{
    return std::min( v.x, v.y );
}

int minimum( const Vector3i& v )
{
    return std::min( v.x, std::min( v.y, v.z ) );
}

int minimum( const Vector4i& v )
{
    return std::min( v.x, std::min( v.y, std::min( v.z, v.w ) ) );
}

float maximum( const Vector2f& v )
{
    return std::max( v.x, v.y );
}

float maximum( const Vector3f& v )
{
    return std::max( v.x, std::max( v.y, v.z ) );
}

float maximum( const Vector4f& v )
{
    return std::max( v.x, std::max( v.y, std::max( v.z, v.w ) ) );
}

int maximum( const Vector2i& v )
{
    return std::max( v.x, v.y );
}

int maximum( const Vector3i& v )
{
    return std::max( v.x, std::max( v.y, v.z ) );
}

int maximum( const Vector4i& v )
{
    return std::max( v.x, std::max( v.y, std::max( v.z, v.w ) ) );
}

Vector2f minimum( const Vector2f& v0, const Vector2f& v1 )
{
    return Vector2f{ std::min( v0.x, v1.x ), std::min( v0.y, v1.y ) };
}

Vector3f minimum( const Vector3f& v0, const Vector3f& v1 )
{
    return Vector3f( std::min( v0.x, v1.x ), std::min( v0.y, v1.y ), std::min( v0.z, v1.z ) );
}

Vector4f minimum( const Vector4f& v0, const Vector4f& v1 )
{
    return Vector4f( std::min( v0.x, v1.x ), std::min( v0.y, v1.y ), std::min( v0.z, v1.z ), std::min( v0.w, v1.w ) );
}

Vector2i minimum( const Vector2i& v0, const Vector2i& v1 )
{
    return{ std::min( v0.x, v1.x ), std::min( v0.y, v1.y ) };
}

Vector3i minimum( const Vector3i& v0, const Vector3i& v1 )
{
    return{ std::min( v0.x, v1.x ), std::min( v0.y, v1.y ), std::min( v0.z, v1.z ) };
}

Vector4i minimum( const Vector4i& v0, const Vector4i& v1 )
{
    return{ std::min( v0.x, v1.x ), std::min( v0.y, v1.y ), std::min( v0.z, v1.z ), std::min( v0.w, v1.w ) };
}

Vector2f maximum( const Vector2f& v0, const Vector2f& v1 )
{
    return{ std::max( v0.x, v1.x ), std::max( v0.y, v1.y ) };
}

Vector3f maximum( const Vector3f& v0, const Vector3f& v1 )
{
    return Vector3f( std::max( v0.x, v1.x ), std::max( v0.y, v1.y ), std::max( v0.z, v1.z ) );
}

Vector4f maximum( const Vector4f& v0, const Vector4f& v1 )
{
    return Vector4f( std::max( v0.x, v1.x ), std::max( v0.y, v1.y ), std::max( v0.z, v1.z ), std::max( v0.w, v1.w ) );
}

Vector2i maximum( const Vector2i& v0, const Vector2i& v1 )
{
    return{ std::max( v0.x, v1.x ), std::max( v0.y, v1.y ) };
}

Vector3i maximum( const Vector3i& v0, const Vector3i& v1 )
{
    return{ std::max( v0.x, v1.x ), std::max( v0.y, v1.y ), std::max( v0.z, v1.z ) };
}

Vector4i maximum( const Vector4i& v0, const Vector4i& v1 )
{
    return{ std::max( v0.x, v1.x ), std::max( v0.y, v1.y ), std::max( v0.z, v1.z ), std::max( v0.w, v1.w ) };
}

float distanceSquared( float x0, float y0, float x1, float y1 )
{
    float dx = x1 - x0;
    float dy = y1 - y0;

    return( dx * dx + dy * dy );
}

float gaussian( float x, float u, float sigma )
{
    float z = sigma * sqrt( TWO_PI );
    float r = x - u;

    float rSquared = r * r;
    float twoSigmaSquared = 2 * sigma * sigma;

    return ( 1.0f / z ) * exp( -rSquared / twoSigmaSquared );
}

} } } // math, core, libcgt
