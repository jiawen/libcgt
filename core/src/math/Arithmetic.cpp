#include <cassert>
#include <cmath>

#include "math/Arithmetic.h"

#include "common/Array1D.h"

namespace libcgt { namespace core { namespace math {

const float s_fReciprocalLog2 = 1.f / logf(2.f);

// Almost .5f = .5f - 1e^(number of exp bits).
const double s_dDoubleMagicRoundEpsilon = 0.5 - 1.4e-11;

const double s_dDoubleMagic = 6755399441055744.0;

int mod( int x, int n )
{
    return ( ( x % n ) + n ) % n;
}

Vector2i mod( const Vector2i& v, const Vector2i& n )
{
    return{ mod( v.x, n.x ), mod( v.y, n.y ) };
}

Vector3i mod( const Vector3i& v, const Vector3i& n )
{
    return
    {
        mod( v.x, n.x ),
        mod( v.y, n.y ),
        mod( v.z, n.z )
    };
}

Vector4i mod( const Vector4i& v, const Vector4i& n )
{
    return
    {
        mod( v.x, n.x ),
        mod( v.y, n.y ),
        mod( v.z, n.z ),
        mod( v.w, n.w )
    };
}

int sign( int x )
{
    if( x < 0 )
    {
        return -1;
    }
    if( x > 0 )
    {
        return 1;
    }
    return 0;
}

int sign( float x )
{
    if( x < 0 )
    {
        return -1;
    }
    if( x > 0 )
    {
        return 1;
    }
    return 0;
}

int sign( double x )
{
    if( x < 0 )
    {
        return -1;
    }
    if( x > 0 )
    {
        return 1;
    }
    return 0;
}

Vector2i sign( const Vector2f& v )
{
    return{ sign( v.x ), sign( v.y ) };
}

Vector3i sign( const Vector3f& v )
{
    return{ sign( v.x ), sign( v.y ), sign( v.z ) };
}

Vector4i sign( const Vector4f& v )
{
    return{ sign( v.x ), sign( v.y ), sign( v.z ), sign( v.w ) };
}

bool sameSign( float x, float y )
{
    return sign( x ) == sign( y );
}

float divideIntsToFloat( int numerator, int denominator )
{
    float fNumerator = static_cast< float >( numerator );
    return fNumerator / denominator;
}

int divideIntsToFloatAndRound( int numerator, int denominator )
{
    float f = divideIntsToFloat( numerator, denominator );
    return roundToInt( f );
}

float percentage( int numerator, int denominator )
{
    return 100.0f * divideIntsToFloat( numerator, denominator );
}

int numBins( int arraySize, int binSize )
{
    return ceilToInt( divideIntsToFloat( arraySize, binSize ) );
}

bool isPowerOfTwo( int x )
{
    if( x <= 0 )
    {
        return 0;
    }
    else
    {
        return( ( x & ( x - 1 ) ) == 0 );
    }
}

int roundToInt( float x )
{
    return static_cast< int >( x + 0.5f );
}

int floatToInt( float x )
{
    return static_cast< int >( x );
}

int floorToInt( float x )
{
    return static_cast< int >( ::floor( x ) );
}

int ceilToInt( float x )
{
    return static_cast< int >( ::ceil( x ) );
}

int roundToInt( double x )
{
    // 2^52 * 1.5, uses limited precision to floor
    x = x + s_dDoubleMagic;
    return( ( int* ) &x )[0];
}

int doubleToInt( double x )
{
    return( ( x < 0 ) ?
        roundToInt( x + s_dDoubleMagicRoundEpsilon ) :
        roundToInt( x - s_dDoubleMagicRoundEpsilon ) );
}

int floorToInt( double x )
{
    return roundToInt( x - s_dDoubleMagicRoundEpsilon );
}

int ceilToInt( double x )
{
    return roundToInt( x + s_dDoubleMagicRoundEpsilon );
}

Vector2f floor( const Vector2f& v )
{
    return Vector2f{ std::floor( v.x ), std::floor( v.y ) };
}

Vector2f ceil( const Vector2f& v )
{
    return Vector2f{ std::ceil( v.x ), std::ceil( v.y ) };
}

Vector2i roundToInt( const Vector2f& v )
{
    return{ roundToInt( v.x ), roundToInt( v.y ) };
}

Vector2i floorToInt( const Vector2f& v )
{
    return{ floorToInt( v.x ), floorToInt( v.y ) };
}

Vector2i ceilToInt( const Vector2f& v )
{
    return{ ceilToInt( v.x ), ceilToInt( v.y ) };
}

Vector3f floor( const Vector3f& v )
{
    return Vector3f( ::floor( v.x ), ::floor( v.y ), ::floor( v.z ) );
}

Vector3f ceil( const Vector3f& v )
{
    return Vector3f( ::ceil( v.x ), ::ceil( v.y ), ::ceil( v.z ) );
}

Vector3i roundToInt( const Vector3f& v )
{
    return{ roundToInt( v.x ), roundToInt( v.y ), roundToInt( v.z ) };
}

Vector3i floorToInt( const Vector3f& v )
{
    return{ floorToInt( v.x ), floorToInt( v.y ), floorToInt( v.z ) };
}

Vector3i ceilToInt( const Vector3f& v )
{
    return{ ceilToInt( v.x ), ceilToInt( v.y ), ceilToInt( v.z ) };
}

Vector4f floor( const Vector4f& v )
{
    return Vector4f( ::floor( v.x ), ::floor( v.y ), ::floor( v.z ), ::floor( v.w ) );
}

Vector4f ceil( const Vector4f& v )
{
    return Vector4f( ::ceil( v.x ), ::ceil( v.y ), ::ceil( v.z ), ::ceil( v.w ) );
}

Vector4i roundToInt( const Vector4f& v )
{
    return{ roundToInt( v.x ), roundToInt( v.y ), roundToInt( v.z ), roundToInt( v.w ) };
}

Vector4i floorToInt( const Vector4f& v )
{
    return{ floorToInt( v.x ), floorToInt( v.y ), floorToInt( v.z ), floorToInt( v.w ) };
}

Vector4i ceilToInt( const Vector4f& v )
{
    return{ ceilToInt( v.x ), ceilToInt( v.y ), ceilToInt( v.z ), ceilToInt( v.w ) };
}

int log2( int x )
{
    int output = 0;
    x >>= 1;
    while( x > 0 )
    {
        ++output;
        x >>= 1;
    }
    return output;
}

float log2( float x )
{
    return( logf( x ) * s_fReciprocalLog2 );
}

int log2ToInt( float x )
{
    return( ( *( int* )( &x ) ) >> 23 ) - 127;
}

uint32_t roundUpToNearestPowerOfTwo( uint32_t v )
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;

    return( v + 1 );
}

int roundUpToNearestMultipleOf4( int x )
{
    return ( x + 3 ) & ( ~( 0x3 ) );
}

int roundUpToNearestMultipleOf8( int x )
{
    return ( x + 7 ) & ( ~( 0x7 ) );
}

int roundUpToNearestMultipleOf16( int x )
{
    return ( x + 15 ) & ( ~( 0xf ) );
}

int roundUpToNearestMultipleOf128( int x )
{
    return ( x + 127 ) & ( ~( 0x7f ) );
}

int roundUpToNearestMultipleOf256( int x )
{
    return ( x + 255 ) & ( ~( 0xff ) );
}

int findNextPerfectSquare( int x )
{
    int y = x;
    while( !isPerfectSquare( y ) )
    {
        ++y;
    }

    return y;
}

int findNextPerfectSquare( int x, int& sqrtOut )
{
    int y = x;
    while( !isPerfectSquare( y, sqrtOut ) )
    {
        ++y;
    }

    return y;
}

bool isPerfectSquare( int x )
{
    int s;
    return isPerfectSquare( x, s );
}

bool isPerfectSquare( int x, int& sqrtOut )
{
    float fx = static_cast< float >( x );
    float sqrtFX = sqrt( fx );
    int sqrtXLower = floorToInt( sqrtFX );
    int sqrtXUpper = sqrtXLower + 1;

    if( sqrtXLower * sqrtXLower == x )
    {
        sqrtOut = sqrtXLower;
        return true;
    }

    if( sqrtXUpper * sqrtXUpper == x )
    {
        sqrtOut = sqrtXUpper;
        return true;
    }

    return false;
}

int integerSquareRoot( int x )
{
    float fx = static_cast< float >( x );
    float sqrtFX = sqrt( fx );
    int sqrtXLower = floorToInt( sqrtFX );
    int sqrtXUpper = sqrtXLower + 1;

    if( sqrtXUpper * sqrtXUpper == x )
    {
        return sqrtXUpper;
    }
    else
    {
        return sqrtXLower;
    }
}

bool inRangeExclusive( float x, float lo, float hi )
{
    return( lo <= x && x < hi );
}

bool inRangeInclusive( float x, float lo, float hi )
{
    return( lo <= x && x <= hi );
}

} } } // math, core, libcgt
