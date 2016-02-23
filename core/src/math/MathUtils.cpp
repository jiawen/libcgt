#include "math/MathUtils.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <limits>

#include "math/Arithmetic.h"

#include <vecmath/Rect2f.h>
#include <vecmath/Rect2i.h>
#include <vecmath/Box3f.h>
#include <vecmath/Box3i.h>

// In Visual Studio, if _MATH_DEFINES_DEFINED is defined
// then M_PI, etc are defined, so just copy them
#ifdef _MATH_DEFINES_DEFINED

const float MathUtils::E = M_E;
const float MathUtils::PI = M_PI;
const float MathUtils::HALF_PI = M_PI_2;
const float MathUtils::QUARTER_PI = M_PI_4;
const float MathUtils::TWO_PI = 2.0f * M_PI;

#else

const float MathUtils::E = 2.71828182845904523536f;
const float MathUtils::PI = 3.14159265358979323846f;
const float MathUtils::HALF_PI = 1.57079632679489661923f;
const float MathUtils::QUARTER_PI = 0.78539816339744830962f;
const float MathUtils::TWO_PI = 2.0f * MathUtils::PI;

#endif

// static
const float MathUtils::PHI = 0.5f * ( 1 + sqrt( 5.0f ) );

// static
const float MathUtils::NEGATIVE_INFINITY = -std::numeric_limits< float >::infinity();

// static
const float MathUtils::POSITIVE_INFINITY = std::numeric_limits< float >::infinity();

//static
bool MathUtils::isNumber( float x )
{
    // See: http://www.johndcook.com/IEEE_exceptions_in_cpp.html
    // returns false if x is NaN
    return( x == x );
}

// static
bool MathUtils::isFinite( float x )
{
    // See: http://www.johndcook.com/IEEE_exceptions_in_cpp.html
    return( x <= FLT_MAX && x >= -FLT_MAX );
}

// static
float MathUtils::cot( float x )
{
    return 1.f / tanf( x );
}

// static
float MathUtils::asinh( float x )
{
    return log(x + sqrt(x * x + 1.f));
}

// static
float MathUtils::degreesToRadians( float degrees )
{
    return static_cast< float >( degrees * MathUtils::PI / 180.0f );
}

// static
double MathUtils::degreesToRadians( double degrees )
{
    return( degrees * MathUtils::PI / 180.0 );
}

// static
float MathUtils::radiansToDegrees( float radians )
{
    return static_cast< float >( radians * 180.0f / MathUtils::PI );
}

// static
double MathUtils::radiansToDegrees( double radians )
{
    return( radians * 180.0 / MathUtils::PI );
}

// static
int MathUtils::clampToRangeExclusive( int x, int lo, int hi )
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
int MathUtils::clampToRangeInclusive( int x, int lo, int hi )
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
Vector2i MathUtils::clampToRectangleExclusive( int x, int y, int left, int bottom, int width, int height )
{
    return
    {
        clampToRangeExclusive( x, left, left + width ),
        clampToRangeExclusive( y, bottom, bottom + height )
    };
}

// static
Vector2i MathUtils::clampToRectangleExclusive( const Vector2i& v, const Vector2i& origin, const Vector2i& size )
{
    return clampToRectangleExclusive( v.x, v.y, origin.x, origin.y, size.x, size.y );
}

// static
Vector2i MathUtils::clampToRectangleExclusive( const Vector2i& v, const Vector2i& size )
{
    return clampToRectangleExclusive( v.x, v.y, 0, 0, size.x, size.y );
}

// static
Vector2i MathUtils::clampToRectangleExclusive( const Vector2i& v, const Rect2i& rect )
{
    return clampToRectangleExclusive( v, rect.origin(), rect.size() );
}

// static
Vector2f MathUtils::clampToRectangle( const Vector2f& v, const Rect2f& rect )
{
    float x = clampToRange( v.x, rect.origin().x, rect.limit().x );
    float y = clampToRange( v.y, rect.origin().y, rect.limit().y );
    return{ x, y };
}

// static
Vector3i MathUtils::clampToBoxExclusive( int x, int y, int z, int left, int bottom, int back, int width, int height, int depth )
{
    return
    {
        clampToRangeExclusive( x, left, left + width ),
        clampToRangeExclusive( y, bottom, bottom + height ),
        clampToRangeExclusive( z, back, back + depth )
    };
}

// static
Vector3i MathUtils::clampToBoxExclusive( const Vector3i& v, const Vector3i& origin, const Vector3i& size )
{
    return clampToBoxExclusive( v.x, v.y, v.z, origin.x, origin.y, origin.z, size.x, size.y, size.z );
}

// static
Vector3i MathUtils::clampToBoxExclusive( const Vector3i& v, const Vector3i& size )
{
    return clampToBoxExclusive( v.x, v.y, v.z, 0, 0, 0, size.x, size.y, size.z );
}

// static
Vector3i MathUtils::clampToBoxExclusive( const Vector3i& v, const Box3i& box )
{
    return clampToBoxExclusive( v, box.origin(), box.size() );
}

// static
Vector3f MathUtils::clampToBox( const Vector3f& v, const Box3f& box )
{
    float x = clampToRange( v.x, box.left(), box.right() );
    float y = clampToRange( v.y, box.bottom(), box.top() );
    float z = clampToRange( v.z, box.back(), box.front() );
    return Vector3f( x, y, z );
}

// static
Vector2f MathUtils::abs( const Vector2f& v )
{
    return{ std::abs( v.x ), std::abs( v.y ) };
}

// static
Vector3f MathUtils::abs( const Vector3f& v )
{
    return Vector3f( std::abs( v.x ), std::abs( v.y ), std::abs( v.z ) );
}

// static
Vector4f MathUtils::abs( const Vector4f& v )
{
    return Vector4f( std::abs( v.x ), std::abs( v.y ), std::abs( v.z ), std::abs( v.w ) );
}

// static
Vector2i MathUtils::abs( const Vector2i& v )
{
    return{ std::abs( v.x ), std::abs( v.y ) };
}

// static
Vector3i MathUtils::abs( const Vector3i& v )
{
    return{ std::abs( v.x ), std::abs( v.y ), std::abs( v.z ) };
}

// static
Vector4i MathUtils::abs( const Vector4i& v )
{
    return{ std::abs( v.x ), std::abs( v.y ), std::abs( v.z ), std::abs( v.w ) };
}

// static
float MathUtils::product( const Vector2f& v )
{
    return v.x * v.y;
}

// static
float MathUtils::product( const Vector3f& v )
{
    return v.x * v.y * v.z;
}

// static
float MathUtils::product( const Vector4f& v )
{
    return v.x * v.y * v.z * v.w;
}

// static
int MathUtils::product( const Vector2i& v )
{
    return v.x * v.y;
}

// static
int MathUtils::product( const Vector3i& v )
{
    return v.x * v.y * v.z;
}

// static
int MathUtils::product( const Vector4i& v )
{
    return v.x * v.y * v.z * v.w;
}

// static
float MathUtils::minimum( const Vector2f& v )
{
    return std::min( v.x, v.y );
}

// static
float MathUtils::minimum( const Vector3f& v )
{
    return std::min( v.x, std::min( v.y, v.z ) );
}

// static
float MathUtils::minimum( const Vector4f& v )
{
    return std::min( v.x, std::min( v.y, std::min( v.z, v.w ) ) );
}

// static
int MathUtils::minimum( const Vector2i& v )
{
    return std::min( v.x, v.y );
}

// static
int MathUtils::minimum( const Vector3i& v )
{
    return std::min( v.x, std::min( v.y, v.z ) );
}

// static
int MathUtils::minimum( const Vector4i& v )
{
    return std::min( v.x, std::min( v.y, std::min( v.z, v.w ) ) );
}

// static
float MathUtils::maximum( const Vector2f& v )
{
    return std::max( v.x, v.y );
}

// static
float MathUtils::maximum( const Vector3f& v )
{
    return std::max( v.x, std::max( v.y, v.z ) );
}

// static
float MathUtils::maximum( const Vector4f& v )
{
    return std::max( v.x, std::max( v.y, std::max( v.z, v.w ) ) );
}

// static
int MathUtils::maximum( const Vector2i& v )
{
    return std::max( v.x, v.y );
}

// static
int MathUtils::maximum( const Vector3i& v )
{
    return std::max( v.x, std::max( v.y, v.z ) );
}

// static
int MathUtils::maximum( const Vector4i& v )
{
    return std::max( v.x, std::max( v.y, std::max( v.z, v.w ) ) );
}

// static
Vector2f MathUtils::minimum( const Vector2f& v0, const Vector2f& v1 )
{
    return Vector2f{ std::min( v0.x, v1.x ), std::min( v0.y, v1.y ) };
}

// static
Vector3f MathUtils::minimum( const Vector3f& v0, const Vector3f& v1 )
{
    return Vector3f( std::min( v0.x, v1.x ), std::min( v0.y, v1.y ), std::min( v0.z, v1.z ) );
}

// static
Vector4f MathUtils::minimum( const Vector4f& v0, const Vector4f& v1 )
{
    return Vector4f( std::min( v0.x, v1.x ), std::min( v0.y, v1.y ), std::min( v0.z, v1.z ), std::min( v0.w, v1.w ) );
}

// static
Vector2i MathUtils::minimum( const Vector2i& v0, const Vector2i& v1 )
{
    return{ std::min( v0.x, v1.x ), std::min( v0.y, v1.y ) };
}

// static
Vector3i MathUtils::minimum( const Vector3i& v0, const Vector3i& v1 )
{
    return{ std::min( v0.x, v1.x ), std::min( v0.y, v1.y ), std::min( v0.z, v1.z ) };
}

// static
Vector4i MathUtils::minimum( const Vector4i& v0, const Vector4i& v1 )
{
    return{ std::min( v0.x, v1.x ), std::min( v0.y, v1.y ), std::min( v0.z, v1.z ), std::min( v0.w, v1.w ) };
}

// static
Vector2f MathUtils::maximum( const Vector2f& v0, const Vector2f& v1 )
{
    return{ std::max( v0.x, v1.x ), std::max( v0.y, v1.y ) };
}

// static
Vector3f MathUtils::maximum( const Vector3f& v0, const Vector3f& v1 )
{
    return Vector3f( std::max( v0.x, v1.x ), std::max( v0.y, v1.y ), std::max( v0.z, v1.z ) );
}

// static
Vector4f MathUtils::maximum( const Vector4f& v0, const Vector4f& v1 )
{
    return Vector4f( std::max( v0.x, v1.x ), std::max( v0.y, v1.y ), std::max( v0.z, v1.z ), std::max( v0.w, v1.w ) );
}

// static
Vector2i MathUtils::maximum( const Vector2i& v0, const Vector2i& v1 )
{
    return{ std::max( v0.x, v1.x ), std::max( v0.y, v1.y ) };
}

// static
Vector3i MathUtils::maximum( const Vector3i& v0, const Vector3i& v1 )
{
    return{ std::max( v0.x, v1.x ), std::max( v0.y, v1.y ), std::max( v0.z, v1.z ) };
}

// static
Vector4i MathUtils::maximum( const Vector4i& v0, const Vector4i& v1 )
{
    return{ std::max( v0.x, v1.x ), std::max( v0.y, v1.y ), std::max( v0.z, v1.z ), std::max( v0.w, v1.w ) };
}

// static
void MathUtils::rescaleRangeToScaleOffset( float inputMin, float inputMax,
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
float MathUtils::rescaleFloatToFloat( float x,
    float inputMin, float inputMax,
    float outputMin, float outputMax )
{
    float inputRange = inputMax - inputMin;
    float outputRange = outputMax - outputMin;

    float fraction = ( x - inputMin ) / inputRange;

    return( outputMin + fraction * outputRange );
}

// static
int MathUtils::rescaleFloatToInt( float x,
    float fMin, float fMax,
    int iMin, int iMax )
{
    float fraction = ( x - fMin ) / ( fMax - fMin );
    return( iMin + Arithmetic::roundToInt( fraction * ( iMax - iMin ) ) );
}

// static
float MathUtils::rescaleIntToFloat( int x,
    int iMin, int iMax,
    float fMin, float fMax )
{
    int inputRange = iMax - iMin;
    float outputRange = fMax - fMin;

    float fraction = static_cast< float >( x - iMin ) / inputRange;
    return( fMin + fraction * outputRange );
}

// static
int MathUtils::rescaleIntToInt( int x,
    int inMin, int inMax,
    int outMin, int outMax )
{
    int inputRange = inMax - inMin;
    int outputRange = outMax - outMin;

    float fraction = static_cast< float >( x - inMin )  / inputRange;
    return Arithmetic::roundToInt( outMin + fraction * outputRange );
}

// static
float MathUtils::distanceSquared( float x0, float y0, float x1, float y1 )
{
    float dx = x1 - x0;
    float dy = y1 - y0;

    return( dx * dx + dy * dy );
}

// static
float MathUtils::gaussian( float x, float u, float sigma )
{
    float z = sigma * sqrt( MathUtils::TWO_PI );
    float r = x - u;

    float rSquared = r * r;
    float twoSigmaSquared = 2 * sigma * sigma;

    return ( 1.0f / z ) * exp( -rSquared / twoSigmaSquared );
}
