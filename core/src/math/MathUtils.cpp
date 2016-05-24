#define _USE_MATH_DEFINES
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <limits>

#include "math/Arithmetic.h"
#include "math/MathUtils.h"

#include <vecmath/Rect2f.h>
#include <vecmath/Rect2i.h>
#include <vecmath/Box3f.h>
#include <vecmath/Box3i.h>

using libcgt::core::math::roundToInt;

namespace libcgt { namespace core { namespace math {

const float E = static_cast< float >( M_E );
const float PI = static_cast< float >( M_PI );
const float HALF_PI = static_cast< float >( M_PI_2 );
const float QUARTER_PI = static_cast< float >( M_PI_4 );
const float TWO_PI = static_cast< float >( 2.0f * M_PI );

// static
const float PHI = 0.5f * ( 1 + sqrt( 5.0f ) );

// static
const float NEGATIVE_INFINITY = -std::numeric_limits< float >::infinity();

// static
const float POSITIVE_INFINITY = std::numeric_limits< float >::infinity();

//static
bool isNumber( float x )
{
    // See: http://www.johndcook.com/IEEE_exceptions_in_cpp.html
    // returns false if x is NaN
    return( x == x );
}

// static
bool isFinite( float x )
{
    // See: http://www.johndcook.com/IEEE_exceptions_in_cpp.html
    return( x <= FLT_MAX && x >= -FLT_MAX );
}

// static
float cot( float x )
{
    return 1.f / tanf( x );
}

// static
float asinh( float x )
{
    return log(x + sqrt(x * x + 1.f));
}

// static
float degreesToRadians( float degrees )
{
    return static_cast< float >( degrees * PI / 180.0f );
}

// static
double degreesToRadians( double degrees )
{
    return( degrees * PI / 180.0 );
}

// static
float radiansToDegrees( float radians )
{
    return static_cast< float >( radians * 180.0f / PI );
}

// static
double radiansToDegrees( double radians )
{
    return( radians * 180.0 / PI );
}

// static
int clampToRangeExclusive( int x, int lo, int hi )
{
    if( x >= hi )
    {
        x = hi - 1;
    }
    if( x < lo )
    {
        x = lo;
    }

    return x;
}

// static
int clampToRangeInclusive( int x, int lo, int hi )
{
    if( x > hi )
    {
        x = hi;
    }
    if( x < lo )
    {
        x = lo;
    }

    return x;
}

// static
Vector2i clampToRectangleExclusive( int x, int y, int left, int bottom, int width, int height )
{
    return
    {
        clampToRangeExclusive( x, left, left + width ),
        clampToRangeExclusive( y, bottom, bottom + height )
    };
}

// static
Vector2i clampToRectangleExclusive( const Vector2i& v, const Vector2i& origin, const Vector2i& size )
{
    return clampToRectangleExclusive( v.x, v.y, origin.x, origin.y, size.x, size.y );
}

// static
Vector2i clampToRectangleExclusive( const Vector2i& v, const Vector2i& size )
{
    return clampToRectangleExclusive( v.x, v.y, 0, 0, size.x, size.y );
}

// static
Vector2i clampToRectangleExclusive( const Vector2i& v, const Rect2i& rect )
{
    return clampToRectangleExclusive( v, rect.origin(), rect.size() );
}

// static
Vector2f clampToRectangle( const Vector2f& v, const Rect2f& rect )
{
    float x = clampToRange( v.x, rect.origin().x, rect.limit().x );
    float y = clampToRange( v.y, rect.origin().y, rect.limit().y );
    return{ x, y };
}

// static
Vector3i clampToBoxExclusive( int x, int y, int z, int left, int bottom, int back, int width, int height, int depth )
{
    return
    {
        clampToRangeExclusive( x, left, left + width ),
        clampToRangeExclusive( y, bottom, bottom + height ),
        clampToRangeExclusive( z, back, back + depth )
    };
}

// static
Vector3i clampToBoxExclusive( const Vector3i& v, const Vector3i& origin, const Vector3i& size )
{
    return clampToBoxExclusive( v.x, v.y, v.z, origin.x, origin.y, origin.z, size.x, size.y, size.z );
}

// static
Vector3i clampToBoxExclusive( const Vector3i& v, const Vector3i& size )
{
    return clampToBoxExclusive( v.x, v.y, v.z, 0, 0, 0, size.x, size.y, size.z );
}

// static
Vector3i clampToBoxExclusive( const Vector3i& v, const Box3i& box )
{
    return clampToBoxExclusive( v, box.origin(), box.size() );
}

// static
Vector3f clampToBox( const Vector3f& v, const Box3f& box )
{
    float x = clampToRange( v.x, box.left(), box.right() );
    float y = clampToRange( v.y, box.bottom(), box.top() );
    float z = clampToRange( v.z, box.back(), box.front() );
    return Vector3f( x, y, z );
}

// static
Vector2f abs( const Vector2f& v )
{
    return{ std::abs( v.x ), std::abs( v.y ) };
}

// static
Vector3f abs( const Vector3f& v )
{
    return Vector3f( std::abs( v.x ), std::abs( v.y ), std::abs( v.z ) );
}

// static
Vector4f abs( const Vector4f& v )
{
    return Vector4f( std::abs( v.x ), std::abs( v.y ), std::abs( v.z ), std::abs( v.w ) );
}

// static
Vector2i abs( const Vector2i& v )
{
    return{ std::abs( v.x ), std::abs( v.y ) };
}

// static
Vector3i abs( const Vector3i& v )
{
    return{ std::abs( v.x ), std::abs( v.y ), std::abs( v.z ) };
}

// static
Vector4i abs( const Vector4i& v )
{
    return{ std::abs( v.x ), std::abs( v.y ), std::abs( v.z ), std::abs( v.w ) };
}

// static
float product( const Vector2f& v )
{
    return v.x * v.y;
}

// static
float product( const Vector3f& v )
{
    return v.x * v.y * v.z;
}

// static
float product( const Vector4f& v )
{
    return v.x * v.y * v.z * v.w;
}

// static
int product( const Vector2i& v )
{
    return v.x * v.y;
}

// static
int product( const Vector3i& v )
{
    return v.x * v.y * v.z;
}

// static
int product( const Vector4i& v )
{
    return v.x * v.y * v.z * v.w;
}

// static
float minimum( const Vector2f& v )
{
    return std::min( v.x, v.y );
}

// static
float minimum( const Vector3f& v )
{
    return std::min( v.x, std::min( v.y, v.z ) );
}

// static
float minimum( const Vector4f& v )
{
    return std::min( v.x, std::min( v.y, std::min( v.z, v.w ) ) );
}

// static
int minimum( const Vector2i& v )
{
    return std::min( v.x, v.y );
}

// static
int minimum( const Vector3i& v )
{
    return std::min( v.x, std::min( v.y, v.z ) );
}

// static
int minimum( const Vector4i& v )
{
    return std::min( v.x, std::min( v.y, std::min( v.z, v.w ) ) );
}

// static
float maximum( const Vector2f& v )
{
    return std::max( v.x, v.y );
}

// static
float maximum( const Vector3f& v )
{
    return std::max( v.x, std::max( v.y, v.z ) );
}

// static
float maximum( const Vector4f& v )
{
    return std::max( v.x, std::max( v.y, std::max( v.z, v.w ) ) );
}

// static
int maximum( const Vector2i& v )
{
    return std::max( v.x, v.y );
}

// static
int maximum( const Vector3i& v )
{
    return std::max( v.x, std::max( v.y, v.z ) );
}

// static
int maximum( const Vector4i& v )
{
    return std::max( v.x, std::max( v.y, std::max( v.z, v.w ) ) );
}

// static
Vector2f minimum( const Vector2f& v0, const Vector2f& v1 )
{
    return Vector2f{ std::min( v0.x, v1.x ), std::min( v0.y, v1.y ) };
}

// static
Vector3f minimum( const Vector3f& v0, const Vector3f& v1 )
{
    return Vector3f( std::min( v0.x, v1.x ), std::min( v0.y, v1.y ), std::min( v0.z, v1.z ) );
}

// static
Vector4f minimum( const Vector4f& v0, const Vector4f& v1 )
{
    return Vector4f( std::min( v0.x, v1.x ), std::min( v0.y, v1.y ), std::min( v0.z, v1.z ), std::min( v0.w, v1.w ) );
}

// static
Vector2i minimum( const Vector2i& v0, const Vector2i& v1 )
{
    return{ std::min( v0.x, v1.x ), std::min( v0.y, v1.y ) };
}

// static
Vector3i minimum( const Vector3i& v0, const Vector3i& v1 )
{
    return{ std::min( v0.x, v1.x ), std::min( v0.y, v1.y ), std::min( v0.z, v1.z ) };
}

// static
Vector4i minimum( const Vector4i& v0, const Vector4i& v1 )
{
    return{ std::min( v0.x, v1.x ), std::min( v0.y, v1.y ), std::min( v0.z, v1.z ), std::min( v0.w, v1.w ) };
}

// static
Vector2f maximum( const Vector2f& v0, const Vector2f& v1 )
{
    return{ std::max( v0.x, v1.x ), std::max( v0.y, v1.y ) };
}

// static
Vector3f maximum( const Vector3f& v0, const Vector3f& v1 )
{
    return Vector3f( std::max( v0.x, v1.x ), std::max( v0.y, v1.y ), std::max( v0.z, v1.z ) );
}

// static
Vector4f maximum( const Vector4f& v0, const Vector4f& v1 )
{
    return Vector4f( std::max( v0.x, v1.x ), std::max( v0.y, v1.y ), std::max( v0.z, v1.z ), std::max( v0.w, v1.w ) );
}

// static
Vector2i maximum( const Vector2i& v0, const Vector2i& v1 )
{
    return{ std::max( v0.x, v1.x ), std::max( v0.y, v1.y ) };
}

// static
Vector3i maximum( const Vector3i& v0, const Vector3i& v1 )
{
    return{ std::max( v0.x, v1.x ), std::max( v0.y, v1.y ), std::max( v0.z, v1.z ) };
}

// static
Vector4i maximum( const Vector4i& v0, const Vector4i& v1 )
{
    return{ std::max( v0.x, v1.x ), std::max( v0.y, v1.y ), std::max( v0.z, v1.z ), std::max( v0.w, v1.w ) };
}

// static
void rescaleRangeToScaleOffset( float inputMin, float inputMax,
    float outputMin, float outputMax,
    float& scale, float& offset )
{
    float inputRange = inputMax - inputMin;
    float outputRange = outputMax - outputMin;

    // y = outputMin + [ ( x - inputMin ) / inputRange ] * outputRange
    //   = outputMin + ( x * outputRange / inputRange ) - ( inputMin * outputRange / inputRange )
    //
    // -->
    //
    // scale = outputRange / inputRange
    // offset = outputMin - inputMin * outputRange / inputRange

    scale = outputRange / inputRange;
    offset = outputMin - inputMin * scale;
}

// static
float rescaleFloatToFloat( float x,
    float inputMin, float inputMax,
    float outputMin, float outputMax )
{
    float inputRange = inputMax - inputMin;
    float outputRange = outputMax - outputMin;

    float fraction = ( x - inputMin ) / inputRange;

    return( outputMin + fraction * outputRange );
}

// static
int rescaleFloatToInt( float x,
    float fMin, float fMax,
    int iMin, int iMax )
{
    float fraction = ( x - fMin ) / ( fMax - fMin );
    return( iMin + roundToInt( fraction * ( iMax - iMin ) ) );
}

// static
float rescaleIntToFloat( int x,
    int iMin, int iMax,
    float fMin, float fMax )
{
    int inputRange = iMax - iMin;
    float outputRange = fMax - fMin;

    float fraction = static_cast< float >( x - iMin ) / inputRange;
    return( fMin + fraction * outputRange );
}

// static
int rescaleIntToInt( int x,
    int inMin, int inMax,
    int outMin, int outMax )
{
    int inputRange = inMax - inMin;
    int outputRange = outMax - outMin;

    float fraction = static_cast< float >( x - inMin )  / inputRange;
    return roundToInt( outMin + fraction * outputRange );
}

// static
float distanceSquared( float x0, float y0, float x1, float y1 )
{
    float dx = x1 - x0;
    float dy = y1 - y0;

    return( dx * dx + dy * dy );
}

// static
float gaussian( float x, float u, float sigma )
{
    float z = sigma * sqrt( TWO_PI );
    float r = x - u;

    float rSquared = r * r;
    float twoSigmaSquared = 2 * sigma * sigma;

    return ( 1.0f / z ) * exp( -rSquared / twoSigmaSquared );
}

} } } // math, core, libcgt
