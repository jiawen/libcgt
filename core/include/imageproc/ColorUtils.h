#pragma once

#include <cstdint>

#include <common/BasicTypes.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>

namespace libcgt
{
namespace core
{
namespace imageproc
{
namespace colorutils
{
    // the epsilon used when converting to the log domain
    // and then input is "luminance" from rgbToLuminance()
    // value is 1 / 256
    const float LOG_LUMINANCE_EPSILON = 0.001f;

    // the epsilon used when converting to the log domain
    // and then input is the L channel from CIE-Lab
    // the value is ( 1 / 256 ) * ( 100 / 256 );
    const float LOG_LAB_EPSILON = 10.0f;

    // Divides each component by 255 and returns them as floats.
    float toFloat( uint8_t x );
    Vector2f toFloat( const uint8x2& v );
    Vector3f toFloat( const uint8x3& v );
    Vector4f toFloat( const uint8x4& v );

    // Converts f in [0,1] to an unsigned byte.
    // Call saturate() first if x is not in [0,1].
    uint8_t toUInt8( float x );
    uint8x2 toUInt8( const Vector2f& v );
    uint8x3 toUInt8( const Vector3f& v );
    uint8x4 toUInt8( const Vector4f& v );

    // Converts a signed byte in [-127,127] to a float in [-1,1].
    // -128 is mapped to -1 as per OpenGL: output = max(x/127, -1).
    float toFloat( int8_t x );
    Vector2f toFloat( const int8x2& v );
    Vector3f toFloat( const int8x3& v );
    Vector4f toFloat( const int8x4& v );

    // Converts x in [-1,1] to a signed byte in [-127,127].
    // Behavior for f outside [-1,1] is undefined.
    int8_t toSInt8( float x );
    int8x2 toSInt8( const Vector2f& v );
    int8x3 toSInt8( const Vector3f& v );
    int8x4 toSInt8( const Vector4f& v );

    // Convert RGB to luminance (gray).
    float rgbToLuminance( const Vector3f& rgb );
    float rgbToLuminance( uint8x3 rgb );

    // Convert RGB to the XYZ color space.
    Vector3f rgb2xyz( const Vector3f& rgb );

    // Convert XYZ to the CIE-Lab color space.
    // xyzRef = float3( 95.047, 100, 108.883 )
    // epsilon = 216 / 24389 = 0.00856
    // kappa = 24389 / 27 = 903.2963
    Vector3f xyz2lab( const Vector3f& xyz,
        const Vector3f& xyzRef = Vector3f( 95.047f, 100.f, 108.883f ),
        float epsilon = 216.f / 24389.f,
        float kappa = 24389.f / 27.f );

    // Convert RGB to the CIE-Lab color space.
    Vector3f rgb2lab( const Vector3f& rgb );

    Vector3f hsv2rgb( const Vector3f& hsv );

    // alpha is preserved
    Vector4f hsva2rgba( const Vector4f& hsva );

    // given x in [0,1], returns an RGBA color like MATLAB's "jet" colormap
    Vector4f colorMapJet( float x );

    // returns the logarithm of the L channel of an Lab image
    // offset by LOG_LAB_EPSILON and rescaled between 0 and 100
    float logL( float l );

    // returns the anti-logarithm of the L channel of an Lab image
    // offset by LOG_LAB_EPSILON and rescaled between 0 and 100
    float expL( float ll );

    // clamps f to [0,1]
    float saturate( float x );
    Vector2f saturate( const Vector2f& v );
    Vector3f saturate( const Vector3f& v );
    Vector4f saturate( const Vector4f& v );
}
}
}
}
