#include "imageproc/ColorUtils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <math/Arithmetic.h>
#include <common/Iterators.h>

float libcgt::core::imageproc::colorutils::toFloat( uint8_t x )
{
    return x / 255.f;
}

Vector2f libcgt::core::imageproc::colorutils::toFloat( const uint8x2& v )
{
    return Vector2f{ toFloat( v.x ), toFloat( v.y ) };
}

Vector3f libcgt::core::imageproc::colorutils::toFloat( const uint8x3& v )
{
    return Vector3f( toFloat( v.x ), toFloat( v.y ), toFloat( v.z ) );
}

Vector4f libcgt::core::imageproc::colorutils::toFloat( const uint8x4& v )
{
    return Vector4f( toFloat( v.x ), toFloat( v.y ), toFloat( v.z ), toFloat( v.w ) );
}

uint8_t libcgt::core::imageproc::colorutils::toUInt8( float x )
{
    return static_cast< uint8_t >( 255.0f * x );
}

uint8x2 libcgt::core::imageproc::colorutils::toUInt8( const Vector2f& v )
{
    return{ toUInt8( v.x ), toUInt8( v.y ) };
}

uint8x3 libcgt::core::imageproc::colorutils::toUInt8( const Vector3f& v )
{
    return{ toUInt8( v.x ), toUInt8( v.y ), toUInt8( v.z ) };
}

uint8x4 libcgt::core::imageproc::colorutils::toUInt8( const Vector4f& v )
{
    return{ toUInt8( v.x ), toUInt8( v.y ), toUInt8( v.z ), toUInt8( v.w ) };
}

float libcgt::core::imageproc::colorutils::toFloat( int8_t x )
{
    return std::max( x / 127.0f, -1.0f );
}

Vector2f libcgt::core::imageproc::colorutils::toFloat( const int8x2& v )
{
    return{ toFloat( v.x ), toFloat( v.y ) };
}

Vector3f libcgt::core::imageproc::colorutils::toFloat( const int8x3& v )
{
    return Vector3f( toFloat( v.x ), toFloat( v.y ), toFloat( v.z ) );
}

Vector4f libcgt::core::imageproc::colorutils::toFloat( const int8x4& v )
{
    return Vector4f( toFloat( v.x ), toFloat( v.y ), toFloat( v.z ), toFloat( v.w ) );
}

int8_t libcgt::core::imageproc::colorutils::toSInt8( float x )
{
    return static_cast< int8_t >( x * 127.0f );
}

int8x2 libcgt::core::imageproc::colorutils::toSInt8( const Vector2f& v )
{
    return{ toSInt8( v.x ), toSInt8( v.y ) };
}

int8x3 libcgt::core::imageproc::colorutils::toSInt8( const Vector3f& v )
{
    return{ toSInt8( v.x ), toSInt8( v.y ), toSInt8( v.z ) };
}

int8x4 libcgt::core::imageproc::colorutils::toSInt8( const Vector4f& v )
{
    return{ toSInt8( v.x ), toSInt8( v.y ), toSInt8( v.z ), toSInt8( v.w ) };
}

float libcgt::core::imageproc::colorutils::rgbToLuminance( const Vector3f& rgb )
{
    return( 0.3279f * rgb.x + 0.6557f * rgb.y + 0.0164f * rgb.z );
}

float libcgt::core::imageproc::colorutils::rgbToLuminance( uint8x3 rgb )
{
    return
    (
        0.3279f * toFloat( rgb.x ) +
        0.6557f * toFloat( rgb.y ) +
        0.0164f * toFloat( rgb.z )
    );
}

Vector3f libcgt::core::imageproc::colorutils::rgb2xyz( const Vector3f& rgb )
{
    float rOut = ( rgb.x > 0.04045f ) ?
        pow( ( rgb.x + 0.055f ) / 1.055f, 2.4f ) :
        rgb.x / 12.92f;
    float gOut = ( rgb.y > 0.04045 ) ?
        pow( ( rgb.y + 0.055f ) / 1.055f, 2.4f ) :
        rgb.y / 12.92f;
    float bOut = ( rgb.z > 0.04045f ) ?
        pow( ( rgb.z + 0.055f ) / 1.055f, 2.4f ) :
        rgb.z / 12.92f;

    Vector3f rgbOut = 100 * Vector3f( rOut, gOut, bOut );

    return Vector3f
    (
        Vector3f::dot( rgbOut, Vector3f( 0.4124f, 0.3576f, 0.1805f ) ),
        Vector3f::dot( rgbOut, Vector3f( 0.2126f, 0.7152f, 0.0722f ) ),
        Vector3f::dot( rgbOut, Vector3f( 0.0193f, 0.1192f, 0.9505f ) )
    );
}

Vector3f libcgt::core::imageproc::colorutils::xyz2lab( const Vector3f& xyz,
                             const Vector3f& xyzRef,
                             float epsilon,
                             float kappa )
{
    Vector3f xyzNormalized = xyz / xyzRef;

    float fx = ( xyzNormalized.x > epsilon ) ?
        pow( xyzNormalized.x, 1.f / 3.f ) :
        ( ( kappa * xyzNormalized.x + 16.f ) / 116.f );
    float fy = ( xyzNormalized.y > epsilon ) ?
        pow( xyzNormalized.y, 1.f / 3.f ) :
        ( ( kappa * xyzNormalized.y + 16.f ) / 116.f );
    float fz = ( xyzNormalized.z > epsilon ) ?
        pow( xyzNormalized.z, 1.f / 3.f ) :
        ( ( kappa * xyzNormalized.z + 16.f ) / 116.f );

    return Vector3f
    (
        ( 116.f * fy ) - 16.f,
        500.f * ( fx - fy ),
        200.f * ( fy - fz )
    );
}

Vector3f libcgt::core::imageproc::colorutils::rgb2lab( const Vector3f& rgb )
{
    return libcgt::core::imageproc::colorutils::xyz2lab( libcgt::core::imageproc::colorutils::rgb2xyz( rgb ) );
}

Vector3f libcgt::core::imageproc::colorutils::hsv2rgb( const Vector3f& hsv )
{
    float h = hsv.x;
    float s = hsv.y;
    float v = hsv.z;

    float r;
    float g;
    float b;

    h *= 360.f;
    int i;
    float f, p, q, t;

    if( s == 0 )
    {
        // achromatic (grey)
        return Vector3f( v, v, v );
    }
    else
    {
        h /= 60.f; // sector 0 to 5
        i = Arithmetic::floorToInt( h );
        f = h - i; // factorial part of h
        p = v * ( 1.f - s );
        q = v * ( 1.f - s * f );
        t = v * ( 1.f - s * ( 1.f - f ) );

        switch( i )
        {
            case 0: r = v; g = t; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = t; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = t; g = p; b = v; break;
            default: r = v; g = p; b = q; break;
        }

        return Vector3f( r, g, b );
    }
}

Vector4f libcgt::core::imageproc::colorutils::hsva2rgba( const Vector4f& hsva )
{
    return Vector4f( hsv2rgb( hsva.xyz ), hsva.w );
}

Vector4f libcgt::core::imageproc::colorutils::colorMapJet( float x )
{
    float fourX = 4 * x;
    float r = std::min( fourX - 1.5f, -fourX + 4.5f );
    float g = std::min( fourX - 0.5f, -fourX + 3.5f );
    float b = std::min( fourX + 0.5f, -fourX + 2.5f );

    return saturate( Vector4f( r, g, b, 1 ) );
}

float libcgt::core::imageproc::colorutils::logL( float l )
{
    const float logMin = log( LOG_LAB_EPSILON );
    const float logRange = log( 100 + LOG_LAB_EPSILON ) - logMin;

    float logL = log( l + LOG_LAB_EPSILON );

    // scale between 0 and 1
    float logL_ZO = ( logL - logMin ) / logRange;

    // scale between 0 and 100
    return 100.f * logL_ZO;
}

float libcgt::core::imageproc::colorutils::expL( float ll )
{
    const float logMin = log( LOG_LAB_EPSILON );
    const float logRange = log( 100 + LOG_LAB_EPSILON ) - logMin;

    // scale between 0 and 1
    float logL_ZO = ll / 100.f;
    // bring back to log scale
    float logL = logL_ZO * logRange + logMin;

    // exponentiate
    return exp( logL ) - LOG_LAB_EPSILON;
}

float libcgt::core::imageproc::colorutils::saturate( float x )
{
    if( x < 0 )
    {
        x = 0;
    }
    if( x > 1 )
    {
        x = 1;
    }
    return x;
}

Vector2f libcgt::core::imageproc::colorutils::saturate( const Vector2f& v )
{
    return
    {
        libcgt::core::imageproc::colorutils::saturate( v[ 0 ] ),
        libcgt::core::imageproc::colorutils::saturate( v[ 1 ] )
    };
}

Vector3f libcgt::core::imageproc::colorutils::saturate( const Vector3f& v )
{
    return Vector3f
    (
        libcgt::core::imageproc::colorutils::saturate( v[ 0 ] ),
        libcgt::core::imageproc::colorutils::saturate( v[ 1 ] ),
        libcgt::core::imageproc::colorutils::saturate( v[ 2 ] )
    );
}

Vector4f libcgt::core::imageproc::colorutils::saturate( const Vector4f& v )
{
    return Vector4f
    (
        libcgt::core::imageproc::colorutils::saturate( v[ 0 ] ),
        libcgt::core::imageproc::colorutils::saturate( v[ 1 ] ),
        libcgt::core::imageproc::colorutils::saturate( v[ 2 ] ),
        libcgt::core::imageproc::colorutils::saturate( v[ 3 ] )
    );
}
