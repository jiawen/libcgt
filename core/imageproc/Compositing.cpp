#include "libcgt/core/imageproc/Compositing.h"

#include "libcgt/core/imageproc/ColorUtils.h"
#include "libcgt/core/vecmath/Vector3f.h"
#include "libcgt/core/vecmath/Vector4f.h"

namespace libcgt { namespace core { namespace imageproc {

void over( Array2DReadView< Vector4f > foreground,
    Array2DReadView< Vector4f > background,
    Array2DWriteView< Vector4f > output )
{
    for( int y = 0; y < foreground.height(); ++y )
    {
        for( int x = 0; x < foreground.width(); ++x )
        {
            Vector4f f = foreground[ { x, y } ];
            Vector4f b = background[ { x, y } ];

            float fa = f.w;
            float ba = b.w;

            Vector3f compositeColor = fa * f.xyz + ( 1.f - fa ) * ( b.xyz );
            float compositeAlpha = fa + ba * ( 1 - fa );

            output[ { x, y } ] = Vector4f( compositeColor, compositeAlpha );
        }
    }
}

void extractBackgroundColor( Array2DReadView< Vector4f > composite,
    Array2DReadView< Vector4f > foreground,
    Array2DWriteView< Vector4f > background )
{
    // red channel:
    // c_r = f_a * f_r + ( 1 - f_a ) * b_r
    // b_r = ( c_r - f_a * f_r ) / ( 1 - f_a )
    //
    // alpha channel:
    // c_a = f_a + b_a * ( 1 - f_a )
    // b_a = ( c_a - f_a ) / ( 1 - f_a )

    for( int y = 0; y < composite.height(); ++y )
    {
        for( int x = 0; x < composite.width(); ++x )
        {
            Vector4f cRGBA = composite[ { x, y } ];
            Vector4f fRGBA = foreground[ { x, y } ];

            Vector4f bRGBA = extractBackgroundColor( cRGBA, fRGBA );
            background[ { x, y } ] = bRGBA;
        }
    }
}

void extractBackgroundColor( Array2DReadView< uint8x4 > composite,
    Array2DReadView< uint8x4 > foreground,
    Array2DWriteView< uint8x4 > background )
{
    for( int y = 0; y < composite.height(); ++y )
    {
        for( int x = 0; x < composite.width(); ++x )
        {
            Vector4f cRGBA = toFloat( composite[ { x, y } ] );
            Vector4f fRGBA = toFloat( foreground[ { x, y } ] );

            Vector4f bRGBA = extractBackgroundColor( cRGBA, fRGBA );
            background[ { x, y } ] = toUInt8( bRGBA );
        }
    }
}

// static
Vector4f extractBackgroundColor( const Vector4f& composite,
    const Vector4f& foreground )
{
    // red channel:
    // c_r = f_a * f_r + ( 1 - f_a ) * b_r
    // b_r = ( c_r - f_a * f_r ) / ( 1 - f_a )
    //
    // alpha channel:
    // c_a = f_a + b_a * ( 1 - f_a )
    // b_a = ( c_a - f_a ) / ( 1 - f_a )

    Vector3f cRGB = composite.xyz;
    Vector3f fRGB = foreground.xyz;

    Vector4f bRGBA;

    float fa = foreground.w;
    if( fa < 1.f )
    {
        float ca = composite.w;

        Vector3f bRGB = ( cRGB - fa * fRGB ) / ( 1.f - fa );
        float ba = ( ca - fa ) / ( 1.f - fa );

#if 0
        if( bRGB.x() < 0 || bRGB.y() < 0 || bRGB.z() < 0 ||
            bRGB.x() > 1 || bRGB.y() > 1 || bRGB.z() > 1 )
        {
            bRGBA = Vector4f( cRGB, 0.f );
        }
        else
        {
            bRGBA = Vector4f( bRGB, ba );
        }
#else
        bRGBA = Vector4f( bRGB, ba );
#endif
    }
    else // foreground alpha = 1, set the background to the composite color with alpha = 0
    {
        // bRGBA = Vector4f( 0.f, 0.f, 0.f, 0.f );
        bRGBA = Vector4f( cRGB, 0.f );
    }

    return bRGBA;
}

} } } // imageproc, core, libcgt
